use photon_rs::channels::{alter_channels, invert, swap_channels};
use photon_rs::monochrome::{grayscale, monochrome};
use photon_rs::pipeline::Pipeline;
use photon_rs::PhotonImage;

fn test_image() -> (Vec<u8>, u32, u32) {
    let width = 4;
    let height = 4;
    let raw_pix = vec![
        134, 122, 131, 255, 131, 131, 139, 255, 135, 134, 137, 255, 138, 134, 130, 255,
        126, 125, 119, 255, 131, 134, 129, 255, 137, 134, 132, 255, 130, 126, 130, 255,
        132, 125, 132, 255, 122, 142, 129, 255, 134, 135, 128, 255, 138, 120, 125, 255,
        125, 134, 110, 255, 121, 122, 137, 255, 141, 140, 141, 255, 125, 144, 120, 255,
    ];

    (raw_pix, width, height)
}

#[test]
fn pipeline_matches_chained_scalar_ops() {
    let (raw_pix, width, height) = test_image();

    let mut scalar_image = PhotonImage::new(raw_pix.clone(), width, height);
    grayscale(&mut scalar_image);
    invert(&mut scalar_image);
    alter_channels(&mut scalar_image, 10, -20, 30);

    let pipeline_image =
        Pipeline::from_photon_image(&PhotonImage::new(raw_pix, width, height))
            .grayscale()
            .invert()
            .alter_channels(10, -20, 30)
            .finish();

    assert_eq!(
        pipeline_image.get_raw_pixels(),
        scalar_image.get_raw_pixels()
    );
}

#[test]
fn pipeline_monochrome_matches_scalar() {
    let (raw_pix, width, height) = test_image();

    let mut scalar_image = PhotonImage::new(raw_pix.clone(), width, height);
    monochrome(&mut scalar_image, 40, 50, 100);

    let pipeline_image =
        Pipeline::from_photon_image(&PhotonImage::new(raw_pix, width, height))
            .monochrome(40, 50, 100)
            .finish();

    assert_eq!(
        pipeline_image.get_raw_pixels(),
        scalar_image.get_raw_pixels()
    );
}

#[test]
fn pipeline_swap_channels_matches_scalar() {
    let (raw_pix, width, height) = test_image();

    let mut scalar_image = PhotonImage::new(raw_pix.clone(), width, height);
    swap_channels(&mut scalar_image, 0, 2);

    let pipeline_image =
        Pipeline::from_photon_image(&PhotonImage::new(raw_pix, width, height))
            .swap_channels(0, 2)
            .finish();

    assert_eq!(
        pipeline_image.get_raw_pixels(),
        scalar_image.get_raw_pixels()
    );
}

#[test]
fn pipeline_preserves_order_around_swap_channels() {
    let (raw_pix, width, height) = test_image();

    let mut scalar_image = PhotonImage::new(raw_pix.clone(), width, height);
    grayscale(&mut scalar_image);
    alter_channels(&mut scalar_image, 10, -20, 30);
    swap_channels(&mut scalar_image, 0, 2);
    invert(&mut scalar_image);

    let pipeline_image =
        Pipeline::from_photon_image(&PhotonImage::new(raw_pix, width, height))
            .grayscale()
            .alter_channels(10, -20, 30)
            .swap_channels(0, 2)
            .invert()
            .finish();

    assert_eq!(
        pipeline_image.get_raw_pixels(),
        scalar_image.get_raw_pixels()
    );
}

#[test]
fn pipeline_long_fusible_chain_matches_scalar() {
    let (raw_pix, width, height) = test_image();

    let mut scalar_image = PhotonImage::new(raw_pix.clone(), width, height);
    grayscale(&mut scalar_image);
    invert(&mut scalar_image);
    alter_channels(&mut scalar_image, 10, -20, 30);
    monochrome(&mut scalar_image, 40, 50, 100);
    invert(&mut scalar_image);
    alter_channels(&mut scalar_image, -5, 25, -10);
    monochrome(&mut scalar_image, 0, 20, 40);
    invert(&mut scalar_image);
    alter_channels(&mut scalar_image, 12, -8, 4);
    monochrome(&mut scalar_image, 5, 15, 25);
    invert(&mut scalar_image);
    alter_channels(&mut scalar_image, -30, 5, 10);

    let pipeline_image =
        Pipeline::from_photon_image(&PhotonImage::new(raw_pix, width, height))
            .grayscale()
            .invert()
            .alter_channels(10, -20, 30)
            .monochrome(40, 50, 100)
            .invert()
            .alter_channels(-5, 25, -10)
            .monochrome(0, 20, 40)
            .invert()
            .alter_channels(12, -8, 4)
            .monochrome(5, 15, 25)
            .invert()
            .alter_channels(-30, 5, 10)
            .finish();

    assert_eq!(
        pipeline_image.get_raw_pixels(),
        scalar_image.get_raw_pixels()
    );
}
