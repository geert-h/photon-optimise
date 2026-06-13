use photon_rs::channels::{alter_channels, invert};
use photon_rs::monochrome::grayscale;
use photon_rs::pipeline::Pipeline;
use photon_rs::PhotonImage;

#[test]
fn pipeline_matches_chained_scalar_ops() {
    let width = 4;
    let height = 4;
    let raw_pix = vec![
        134, 122, 131, 255, 131, 131, 139, 255, 135, 134, 137, 255, 138, 134, 130, 255,
        126, 125, 119, 255, 131, 134, 129, 255, 137, 134, 132, 255, 130, 126, 130, 255,
        132, 125, 132, 255, 122, 142, 129, 255, 134, 135, 128, 255, 138, 120, 125, 255,
        125, 134, 110, 255, 121, 122, 137, 255, 141, 140, 141, 255, 125, 144, 120, 255,
    ];

    let mut scalar_image = PhotonImage::new(raw_pix.clone(), width, height);
    grayscale(&mut scalar_image);
    invert(&mut scalar_image);
    alter_channels(&mut scalar_image, 10, -20, 30);

    let pipeline_image =
        Pipeline::from_photon_image(&PhotonImage::new(raw_pix, width, height))
            .gray_scale()
            .invert()
            .alter_channels(10, -20, 30)
            .finish();

    assert_eq!(
        pipeline_image.get_raw_pixels(),
        scalar_image.get_raw_pixels()
    );
}
