from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))

from difix3d_service import difix_infer, load_image


def main() -> None:
	input_image = load_image(ROOT_DIR / "difix3d" / "assets" / "example_input.png")
	ref_image = load_image(ROOT_DIR / "difix3d" / "assets" / "example_ref.png")
	prompt = "remove degradation"

	output_image = difix_infer(
		input_image,
		ref_image,
		prompt,
		num_inference_steps=1,
		timesteps=[199],
		guidance_scale=0.0,
	)
	output_image.save(ROOT_DIR / "example_output.png")


if __name__ == "__main__":
	main()