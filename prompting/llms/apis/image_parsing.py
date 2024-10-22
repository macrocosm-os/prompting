# This file is probably not needed for the project but came bundled with the LLM wrapper I wrote
# for a previous project and thought it may be useful in the future if we expand the project to
# also deal with multimodel inputs.
from fastapi import UploadFile, HTTPException
from PIL import Image
import io
from loguru import logger
import base64


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def encode_image_from_memory(image):
    # Convert the NumPy array to a PIL Image object
    # image = Image.fromarray(image_array.astype('uint8'))

    # Create a BytesIO object to hold the image data
    with io.BytesIO() as image_buffer:
        # Save the image to the buffer in PNG format
        image.save(image_buffer, format="PNG")

        # Move to the beginning of the BytesIO buffer
        image_buffer.seek(0)

        # Encode the image data in base64
        base64_encoded_image = base64.b64encode(image_buffer.read()).decode("utf-8")

    return base64_encoded_image


async def parse_api_image(file: UploadFile) -> Image:
    """
    Parses the API image file and returns an Image object.

    Args:
        file (UploadFile): The image file to be parsed.

    Returns:
        Image: The parsed Image object.

    Raises:
        HTTPException: If the file is not an image or if an error occurs while processing the image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(422, f"File must be an image, received {file.content_type.split('/')[-1]}")
    try:
        contents = await file.read()
        return Image.open(io.BytesIO(contents))

    except Exception as ex:
        logger.exception(ex)
        raise HTTPException(
            500,
            "Something went wrong while processing the image",
        )
