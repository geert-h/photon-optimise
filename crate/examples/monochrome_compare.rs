extern crate photon_rs;

use photon_rs::monochrome::{monochrome, monochrome_scalar};
use photon_rs::native::{open_image, save_image};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_name = "crate/examples/input_images/underground.jpg";

    let mut scalar_img = open_image(file_name)?;
    let mut dispatch_img = scalar_img.clone();

    monochrome_scalar(&mut scalar_img, 40, 50, 100);
    monochrome(&mut dispatch_img, 40, 50, 100);

    save_image(scalar_img.clone(), "output_monochrome_scalar.jpg")?;
    save_image(dispatch_img.clone(), "output_monochrome_dispatch.jpg")?;

    if scalar_img.get_raw_pixels() == dispatch_img.get_raw_pixels() {
        println!("Outputs are byte-identical.");
    } else {
        println!("Outputs differ.");
    }

    println!("Wrote output_monochrome_scalar.jpg");
    println!("Wrote output_monochrome_dispatch.jpg");

    Ok(())
}
