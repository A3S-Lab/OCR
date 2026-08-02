#[cfg(any(feature = "ppocr-v6", feature = "unlimited-ocr"))]
use a3s_use_ocr::OcrProvider;

#[cfg(feature = "ppocr-v6")]
use a3s_use_ocr::{PpOcrV6Provider, PP_OCR_V6_PROVIDER_ID};

#[cfg(feature = "unlimited-ocr")]
use a3s_use_ocr::{
    UnlimitedOcrConfig, UnlimitedOcrProvider, UNLIMITED_OCR_MODEL, UNLIMITED_OCR_PROVIDER_ID,
};

#[cfg(feature = "ppocr-v6")]
#[test]
fn pp_ocr_v6_is_one_provider_behind_the_public_interface() {
    let provider = PpOcrV6Provider::from_env().unwrap();
    let descriptor = provider.descriptor();
    assert_eq!(descriptor.id, PP_OCR_V6_PROVIDER_ID);
    assert_eq!(descriptor.engine, "a3s-power-native");
    assert!(!descriptor.sends_source_off_device);
}

#[cfg(feature = "unlimited-ocr")]
#[test]
fn unlimited_ocr_is_one_provider_behind_the_public_interface() {
    let config = UnlimitedOcrConfig::local("http://127.0.0.1:8000/v1").unwrap();
    let provider = UnlimitedOcrProvider::new(config).unwrap();
    let descriptor = provider.descriptor();
    assert_eq!(descriptor.id, UNLIMITED_OCR_PROVIDER_ID);
    assert_eq!(descriptor.engine, "vllm-openai");
    assert!(!descriptor.sends_source_off_device);
    assert_eq!(provider.config().model(), UNLIMITED_OCR_MODEL);
}
