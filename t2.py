# USAGE
# python easy_ocr.py --image images/arabic_sign.jpg --langs en,ar

# import the necessary packages
from easyocr import Reader
import argparse
import cv2
from matplotlib import pyplot as plt

def plt_imshow(title, image):
  # convert the image frame BGR to RGB color space and display it
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  plt.imshow(image)
  plt.title(title)
  plt.grid(False)
  plt.show()


def cleanup_text(text):
	# strip out non-ASCII text so we can draw the text on the image
	# using OpenCV
	return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# construct the argument parser and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
# 	help="path to input image to be OCR'd")
# ap.add_argument("-l", "--langs", type=str, default="en",
# 	help="comma separated list of languages to OCR")
# ap.add_argument("-g", "--gpu", type=int, default=-1,
# 	help="whether or not GPU should be used")
# args = vars(ap.parse_args())
args = {
  "image": "img/img.png",
  "langs": "en,ru",
  "gpu": -1
}

# break the input languages into a comma separated list
langs = args["langs"].split(",")
print("[INFO] OCR'ing with the following languages: {}".format(langs))

# load the input image from disk
image = cv2.imread(args["image"])

# OCR the input image using EasyOCR
print("[INFO] OCR'ing input image...")
reader = Reader(langs, gpu=args["gpu"] > 0)
results = reader.readtext(image)

# loop over the results
for (bbox, text, prob) in results:
	# display the OCR'd text and associated probability
	print("[INFO] {:.4f}: {}".format(prob, text))

	# unpack the bounding box
	(tl, tr, br, bl) = bbox
	tl = (int(tl[0]), int(tl[1]))
	tr = (int(tr[0]), int(tr[1]))
	br = (int(br[0]), int(br[1]))
	bl = (int(bl[0]), int(bl[1]))

	width = tr[0] - tl[0]
	height = tr[1] - br[1]

	tr0 = tr[0] + width*0.05
	tl0 = tl[0] - width*0.05

	br0 = br[0] + width * 0.05
	bl0 = bl[0] - width * 0.05

	tr1 = tr[1] + height * 0.05
	tl1 = tl[1] + height * 0.05

	br1 = br[1] - height * 0.05
	bl1 = bl[1] - height * 0.05

	tl = (int(tl0), int(tl1))
	tr = (int(tr0), int(tr1))
	br = (int(br0), int(br1))
	bl = (int(bl0), int(bl1))

	# cleanup the text and draw the box surrounding the text along
	# with the OCR'd text itself
	text = cleanup_text(text)
	cv2.rectangle(image, tl, br, (0, 255, 0), 2)
	cv2.putText(image, text, (tl[0], tl[1] - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

plt_imshow("Image", image)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)