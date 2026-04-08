#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include "driver/i2s.h"

static const uint32_t SAMPLE_RATE_HZ = 16000;

static const int I2S_PIN_SCK = 14;
static const int I2S_PIN_WS = 15;
static const int I2S_PIN_SD = 32;

static const char *SERVICE_UUID = "c0de0001-9a5b-4fcd-9f28-5f7d2b4d2a01";
static const char *CHAR_UUID = "c0de0002-9a5b-4fcd-9f28-5f7d2b4d2a01";

static const size_t NOTIFY_BYTES = 240;

static BLECharacteristic *gAudioChar = nullptr;
static volatile bool gDeviceConnected = false;

static uint8_t gNotifyBuf[NOTIFY_BYTES];
static size_t gNotifyIndex = 0;

static int32_t gDcEstimate = 0;

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *pServer) override {
    gDeviceConnected = true;
    Serial.println("BLE Client Connected");
  }

  void onDisconnect(BLEServer *pServer) override {
    gDeviceConnected = false;
    Serial.println("BLE Client Disconnected");
    BLEDevice::startAdvertising();
  }
};

static void setupI2S() {
  i2s_config_t i2s_config = {};
  i2s_config.mode = static_cast<i2s_mode_t>(I2S_MODE_MASTER | I2S_MODE_RX);
  i2s_config.sample_rate = SAMPLE_RATE_HZ;
  i2s_config.bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT;
  i2s_config.channel_format = I2S_CHANNEL_FMT_ONLY_LEFT;
  i2s_config.communication_format = static_cast<i2s_comm_format_t>(I2S_COMM_FORMAT_I2S | I2S_COMM_FORMAT_I2S_MSB);
  i2s_config.intr_alloc_flags = ESP_INTR_FLAG_LEVEL1;
  i2s_config.dma_buf_count = 8;
  i2s_config.dma_buf_len = 256;
  i2s_config.use_apll = false;
  i2s_config.tx_desc_auto_clear = false;
  i2s_config.fixed_mclk = 0;

  i2s_pin_config_t pin_config = {};
  pin_config.bck_io_num = I2S_PIN_SCK;
  pin_config.ws_io_num = I2S_PIN_WS;
  pin_config.data_out_num = I2S_PIN_NO_CHANGE;
  pin_config.data_in_num = I2S_PIN_SD;

  ESP_ERROR_CHECK(i2s_driver_install(I2S_NUM_0, &i2s_config, 0, nullptr));
  ESP_ERROR_CHECK(i2s_set_pin(I2S_NUM_0, &pin_config));
  ESP_ERROR_CHECK(i2s_zero_dma_buffer(I2S_NUM_0));
}

static void setupBLE() {
  BLEDevice::init("AI_Stethoscope");
  BLEDevice::setMTU(247);

  BLEServer *server = BLEDevice::createServer();
  server->setCallbacks(new MyServerCallbacks());

  BLEService *service = server->createService(SERVICE_UUID);

  gAudioChar = service->createCharacteristic(CHAR_UUID, BLECharacteristic::PROPERTY_NOTIFY);
  gAudioChar->addDescriptor(new BLE2902());

  service->start();

  BLEAdvertising *advertising = BLEDevice::getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMinPreferred(0x12);
  BLEDevice::startAdvertising();

  Serial.println("BLE Advertising Started");
}

static inline int16_t clip16(int32_t v) {
  if (v > 32767) return 32767;
  if (v < -32768) return -32768;
  return static_cast<int16_t>(v);
}

static inline int16_t i2s32To16(int32_t sample32) {
  return static_cast<int16_t>(sample32 >> 16);
}

static inline int16_t dcRemove(int16_t sample16) {
  int32_t sample = static_cast<int32_t>(sample16);
  gDcEstimate += (sample - gDcEstimate) >> 8;
  return clip16(sample - gDcEstimate);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("Booting...");
  setupI2S();
  setupBLE();
}

void loop() {
  if (!gDeviceConnected) {
    delay(30);
    return;
  }

  static uint32_t lastStreamLogMs = 0;
  static int32_t i2sSamples[256];
  size_t bytesRead = 0;
  esp_err_t err = i2s_read(I2S_NUM_0, i2sSamples, sizeof(i2sSamples), &bytesRead, portMAX_DELAY);
  if (err != ESP_OK || bytesRead == 0) {
    return;
  }

  if (millis() - lastStreamLogMs > 1000) {
    Serial.println("Streaming...");
    lastStreamLogMs = millis();
  }

  size_t sampleCount = bytesRead / sizeof(int32_t);
  for (size_t i = 0; i < sampleCount; i++) {
    int16_t s16 = i2s32To16(i2sSamples[i]);
    s16 = dcRemove(s16);

    gNotifyBuf[gNotifyIndex++] = static_cast<uint8_t>(s16 & 0xFF);
    gNotifyBuf[gNotifyIndex++] = static_cast<uint8_t>((s16 >> 8) & 0xFF);

    if (gNotifyIndex >= NOTIFY_BYTES) {
      gAudioChar->setValue(gNotifyBuf, NOTIFY_BYTES);
      gAudioChar->notify();
      delay(2);
      gNotifyIndex = 0;
    }
  }
}

