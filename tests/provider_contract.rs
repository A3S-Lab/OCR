use a3s_use_ocr::{OcrProvider, PpOcrV6Provider, PP_OCR_V6_PROVIDER_ID};

#[test]
fn pp_ocr_v6_is_one_provider_behind_the_public_interface() {
    let provider = PpOcrV6Provider::from_env().unwrap();
    let descriptor = provider.descriptor();
    assert_eq!(descriptor.id, PP_OCR_V6_PROVIDER_ID);
    assert_eq!(descriptor.engine, "onnx-runtime");
    assert!(!descriptor.sends_source_off_device);
}
