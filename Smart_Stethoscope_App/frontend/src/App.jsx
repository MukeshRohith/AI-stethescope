import './App.css'

import { useEffect, useMemo, useRef, useState } from 'react'

const SERVICE_UUID = 'c0de0001-9a5b-4fcd-9f28-5f7d2b4d2a01'
const CHAR_UUID = 'c0de0002-9a5b-4fcd-9f28-5f7d2b4d2a01'

const SAMPLE_RATE_HZ = 16000
const AUTO_STOP_BYTES = 320000

function concatUint8(chunks) {
  let total = 0
  for (const c of chunks) total += c.byteLength
  const out = new Uint8Array(total)
  let offset = 0
  for (const c of chunks) {
    out.set(c, offset)
    offset += c.byteLength
  }
  return out
}

function encodeWavPcm16Mono(pcm16leBytes, sampleRateHz) {
  const pcmBytes = pcm16leBytes.byteLength - (pcm16leBytes.byteLength % 2)
  const headerBytes = 44
  const wavBuffer = new ArrayBuffer(headerBytes + pcmBytes)
  const view = new DataView(wavBuffer)

  const writeAscii = (offset, text) => {
    for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i))
  }

  writeAscii(0, 'RIFF')
  view.setUint32(4, headerBytes + pcmBytes - 8, true)
  writeAscii(8, 'WAVE')
  writeAscii(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRateHz, true)
  view.setUint32(28, sampleRateHz * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeAscii(36, 'data')
  view.setUint32(40, pcmBytes, true)

  new Uint8Array(wavBuffer, headerBytes, pcmBytes).set(pcm16leBytes.subarray(0, pcmBytes))
  return new Blob([wavBuffer], { type: 'audio/wav' })
}

function App() {
  const bluetoothAvailable = useMemo(
    () => typeof navigator !== 'undefined' && !!navigator.bluetooth,
    [],
  )

  const [deviceName, setDeviceName] = useState('')
  const [status, setStatus] = useState('Idle')
  const [isConnected, setIsConnected] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [bytesReceived, setBytesReceived] = useState(0)
  const [audioUrl, setAudioUrl] = useState('')
  const [predictionResult, setPredictionResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const deviceRef = useRef(null)
  const characteristicRef = useRef(null)
  const recordingRef = useRef(false)
  const chunksRef = useRef([])
  const totalBytesRef = useRef(0)
  const stopInProgressRef = useRef(false)
  const onValueRef = useRef(null)

  useEffect(() => {
    recordingRef.current = isRecording
  }, [isRecording])

  useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl)
      const dev = deviceRef.current
      if (dev?.gatt?.connected) dev.gatt.disconnect()
    }
  }, [audioUrl])

  const connect = async () => {
    setStatus('Requesting device...')
    setAudioUrl('')
    setBytesReceived(0)

    const device = await navigator.bluetooth.requestDevice({
      filters: [{ name: 'AI_Stethoscope' }],
      optionalServices: [SERVICE_UUID],
    })

    deviceRef.current = device
    setDeviceName(device.name || 'ESP32')

    device.addEventListener('gattserverdisconnected', () => {
      setIsConnected(false)
      setIsRecording(false)
      characteristicRef.current = null
      setStatus('Disconnected')
    })

    setStatus('Connecting...')
    const server = await device.gatt.connect()
    const service = await server.getPrimaryService(SERVICE_UUID)
    const characteristic = await service.getCharacteristic(CHAR_UUID)
    characteristicRef.current = characteristic

    const onValue = (event) => {
      if (!recordingRef.current) return
      const dv = event.target.value
      const chunk = new Uint8Array(dv.buffer.slice(dv.byteOffset, dv.byteOffset + dv.byteLength))
      if (chunk.byteLength === 0) return
      chunksRef.current.push(chunk)
      setBytesReceived((n) => n + chunk.byteLength)
      totalBytesRef.current += chunk.byteLength
      if (totalBytesRef.current >= AUTO_STOP_BYTES) {
        stopRecording()
      }
    }
    onValueRef.current = onValue
    characteristic.addEventListener('characteristicvaluechanged', onValue)

    setIsConnected(true)
    setStatus('Connected')
  }

  const disconnect = async () => {
    const characteristic = characteristicRef.current
    if (characteristic && onValueRef.current) {
      characteristic.removeEventListener('characteristicvaluechanged', onValueRef.current)
      onValueRef.current = null
    }

    const dev = deviceRef.current
    if (dev?.gatt?.connected) dev.gatt.disconnect()
  }

  const stopRecording = async () => {
    if (stopInProgressRef.current) return
    stopInProgressRef.current = true

    const characteristic = characteristicRef.current
    if (!characteristic) {
      stopInProgressRef.current = false
      return
    }

    setStatus('Stopping...')
    await characteristic.stopNotifications().catch(() => null)
    setIsRecording(false)

    const pcmBytes = concatUint8(chunksRef.current)
    chunksRef.current = []

    const wavBlob = encodeWavPcm16Mono(pcmBytes, SAMPLE_RATE_HZ)
    const url = URL.createObjectURL(wavBlob)
    setAudioUrl(url)

    setIsAnalyzing(true)
    setPredictionResult(null)
    try {
      const formData = new FormData()
      formData.append('file', wavBlob, 'steth_recording.wav')
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 60000)
      const res = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      clearTimeout(timeoutId)
      if (!res.ok) {
        const text = await res.text().catch(() => '')
        throw new Error(text || `AI request failed (${res.status})`)
      }
      const json = await res.json()
      setPredictionResult(json)
    } catch (e) {
      const msg =
        e && typeof e === 'object' && 'name' in e && e.name === 'AbortError'
          ? 'AI request timed out'
          : e instanceof Error
            ? e.message
            : 'AI request failed'
      setPredictionResult({ error: msg })
    } finally {
      setIsAnalyzing(false)
    }

    setStatus('Ready')
  }

  const startRecording = async () => {
    const characteristic = characteristicRef.current
    if (!characteristic) return

    if (audioUrl) URL.revokeObjectURL(audioUrl)
    setAudioUrl('')
    setBytesReceived(0)
    chunksRef.current = []
    totalBytesRef.current = 0
    stopInProgressRef.current = false
    setPredictionResult(null)

    setStatus('Recording...')
    setIsRecording(true)
    await characteristic.startNotifications()
  }

  return (
    <>
      <main className="page">
        <header className="topbar">
          <div>
            <h1>AI Stethoscope Bridge</h1>
            <p className="sub">
              Web Bluetooth demo: BLE notifications → WAV (mono, 16kHz, 16-bit)
            </p>
          </div>
          <div className="pill">
            <span className={`dot ${isConnected ? 'ok' : ''}`}></span>
            <span>{isConnected ? `Connected${deviceName ? `: ${deviceName}` : ''}` : 'Not connected'}</span>
          </div>
        </header>

        {!bluetoothAvailable ? (
          <section className="card">
            <h2>Web Bluetooth not available</h2>
            <p className="muted">
              Use a Chromium-based browser. Web Bluetooth typically requires HTTPS or localhost.
            </p>
          </section>
        ) : (
          <>
            <section className="grid">
              <div className="card">
                <h2>Connection</h2>
                <div className="row">
                  <button className="btn" onClick={connect} disabled={isConnected}>
                    Connect to Stethoscope
                  </button>
                  <button className="btn secondary" onClick={disconnect} disabled={!isConnected}>
                    Disconnect
                  </button>
                </div>
                <div className="kv">
                  <div className="k">Status</div>
                  <div className="v">{status}</div>
                  <div className="k">Service UUID</div>
                  <div className="v mono">{SERVICE_UUID}</div>
                  <div className="k">Characteristic UUID</div>
                  <div className="v mono">{CHAR_UUID}</div>
                </div>
              </div>

              <div className="card">
                <h2>Recording</h2>
                <div className="row">
                  {!isRecording ? (
                    <button className="btn primary" onClick={startRecording} disabled={!isConnected}>
                      Start Recording (10s Auto-Stop)
                    </button>
                  ) : (
                    <button className="btn danger" onClick={stopRecording}>
                      Stop Recording
                    </button>
                  )}
                </div>
                <div className="rec">
                  <div className={`wave ${isRecording ? 'on' : ''}`} aria-hidden="true">
                    <svg viewBox="0 0 200 40" preserveAspectRatio="none">
                      <path d="M0 20 C 20 5, 40 35, 60 20 S 100 35, 120 20 S 160 5, 200 20" />
                    </svg>
                  </div>
                  <div className="recMeta">
                    <div className="kv compact">
                      <div className="k">Sample Rate</div>
                      <div className="v">{SAMPLE_RATE_HZ} Hz</div>
                      <div className="k">Received</div>
                      <div className="v">{bytesReceived.toLocaleString()} bytes</div>
                    </div>
                  </div>
                </div>
                <p className="muted">Stop recording to export a WAV file.</p>
              </div>
            </section>

            <section className="card">
              <h2>Playback</h2>
              {audioUrl ? (
                <>
                  <audio controls src={audioUrl} />
                  <div className="row">
                    <a className="btn secondary" href={audioUrl} download="steth_recording.wav">
                      Download WAV
                    </a>
                  </div>
                </>
              ) : (
                <p className="muted">No recording yet.</p>
              )}
            </section>

            <section className="card">
              <h2>AI Analysis</h2>
              {isAnalyzing ? (
                <p className="muted">Analyzing...</p>
              ) : predictionResult ? (
                predictionResult.error ? (
                  <p className="muted">{predictionResult.error}</p>
                ) : (
                  <div className="kv">
                    <div className="k">Prediction</div>
                    <div className="v">{String(predictionResult.prediction)}</div>
                    <div className="k">Diagnosis</div>
                    <div className="v">{String(predictionResult.diagnosis)}</div>
                    <div className="k">Probability</div>
                    <div className="v">{String(predictionResult.probability)}</div>
                  </div>
                )
              ) : (
                <p className="muted">No analysis yet.</p>
              )}
            </section>
          </>
        )}
      </main>
    </>
  )
}

export default App
