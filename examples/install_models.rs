use a3s_use_ocr::install_ppocr_v6;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let status = install_ppocr_v6(true).await?;
    println!("{}", serde_json::to_string_pretty(&status)?);
    Ok(())
}
