from SLIC import segmentImage

class ProcessedRegion:

	def __init__(self):
		i = 123

	def hello(self):
		return 'hello World'



class FullImage:


	def __init__(self, filename):
		self.filename = filename


	def getFile(self):
		return self.filename





if __name__=='__main__':
	imageFile ='dog2.jpg'
	numSegments = 100

	image, segments = segmentImage(imageFile, numSegments)
