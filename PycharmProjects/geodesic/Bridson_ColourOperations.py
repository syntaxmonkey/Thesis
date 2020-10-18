
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000


def diffCIEColours(cie1, cie2):
	return delta_e_cie2000(cie1, cie2)


def convertRGB_CIE(rgb):
	rgb1 = sRGBColor( rgb[0], rgb[1], rgb[2], True)
	cie = convert_color(rgb1, LabColor)
	return cie

if __name__ == '__main__':
	rgb1 = sRGBColor(255, 255, 0, True)
	cie1 = convert_color(rgb1, LabColor)
	print(rgb1, "-->", cie1)

	rgb2 = sRGBColor(0, 255, 0, True)
	cie2 = convert_color(rgb2, LabColor)
	print(rgb2, "-->", cie2)

	rgb3 = sRGBColor(0, 0, 255, True)
	cie3 = convert_color(rgb3, LabColor)
	print(rgb3, "-->", cie3)


	print("rgb1 --> rgb2", delta_e_cie2000(cie1, cie2))
	print("rgb1 --> rgb3", delta_e_cie2000(cie1, cie3))
	print("rgb2 --> rgb3", delta_e_cie2000(cie2, cie3))
	print("rgb3 --> rgb1", delta_e_cie2000(cie3, cie1))
	print("rgb3 --> rgb2", delta_e_cie2000(cie3, cie2))
