import numpy as np
import math
import Bridson_Common



class LineEndPointIndex:
	def __init__(self):
		self.__lineEndPointIndex = {}
		pass


	def setLineIndeces(self, line, startIndex, endIndex):
		lineTuple = Bridson_Common.convertToTuple(line)

		lineStartEndIndeces = [startIndex, endIndex]
		self.__lineEndPointIndex[lineTuple] = lineStartEndIndeces
		# if lineTuple in self.__lineEndPointIndex:
		# 	lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		# else:
		# 	self.__lineEndPointIndex[ lineTuple ] = lineStartEndIndeces

		pass

	def updateLineStartIndex(self, line, startIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]

		if lineStartEndIndeces[0] > 0:
			lineStartEndIndeces[0] = startIndex
		pass


	def updateLineEndIndex(self, line, endIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]

		if lineStartEndIndeces[1] > 0:
			lineStartEndIndeces[1] = endIndex
		pass


	def updateLineCollisionStartIndex(self, line):
		lineTuple = Bridson_Common.convertToTuple(line)
		lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces[0] = 0 - abs( lineStartEndIndeces[0] )


	def updateLineCollisionEndIndex(self, line):
		lineTuple = Bridson_Common.convertToTuple(line)
		lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces[1] = 0 - abs( lineStartEndIndeces[1] )



	def getLineEndIndex(self, line):
		lineTuple = Bridson_Common.convertToTuple(line)
		return self.__lineEndPointIndex[ lineTuple ]


	def isLineBounded(self, line):
		'''

		:param line:
		:return: True if both indeces are negative.
		'''
		lineTuple = Bridson_Common.convertToTuple(line)
		lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		if lineStartEndIndeces[0] < 0 and lineStartEndIndeces[1] < 0:
			return True
		return False


	def realignIndexes(self):
		# Iterate through all the indeces.  Convert any negative values to positive.
		for indeces in self.__lineEndPointIndex.values():
			indeces[0] = abs(indeces[0])
			indeces[1] = abs(indeces[1])

class GridCells:
	def __init__(self, LimitX, LimitY, cellsize):
		self.LimitX = LimitX
		self.LimitY = LimitY
		self.cellsize = cellsize

		self.segmentToCell = {}
		self.cellContent = {} # Cell content is a list of tuples. The tuples are (segment, lineTuple).
		self.lineStartEndIndex = {}

		self.pessimisticCheck = True
		pass


	def addLine(self, line, startIndex, endIndex):
		## working for adding whole line.
		## working for partial lines.
		'''
		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
		Create an entry for (start,end) in segmentToCell.
		The cells will be based on start and end coordinates.
		'''
		tupleLine = Bridson_Common.convertToTuple(line)
		# Iterate through all the sub-segments.
		for index in range(startIndex, len(line) - endIndex - 1):
		# for index in range(len(line) - 1):
			segment = (line[index], line[index+1])
			self.__addSegmentToCellMapping(segment, tupleLine)
		# self.lineStartEndIndex[ Bridson_Common.convertToTuple( line ) ] = (startIndex, endIndex)
		pass


	def addSegment(self, segments, line):
		'''
		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
		Create an entry for (start,end) in segmentToCell.
		The cells will be based on start and end coordinates.
		'''
		lineTuple = Bridson_Common.convertToTuple( line )
		# Iterate through all the sub-segments.
		for index in range( len(segments) - 1 ):
			segment = (segments[index], segments[index+1])
			print("addSegment:", segment)
			self.__addSegmentToCellMapping(segment, lineTuple)
		pass




	def detectCollision(self, segment, line):
		'''

		:param segment:
		:param line:
		:return: True if there is a collision, False otherwise.
		'''
		'''
		Detect if there is a collision by adding this segment.
		'''
		print("Detecting Collision for segment:", Bridson_Common.convertToTuple( segment ) )
		# Find which cell it belongs to.
		cellContent0 = self.getCellContent( segment[0] )
		cellContent1 = self.getCellContent( segment[1] )
		# Check for collision with segments in the cell.

		collisionDetected = False
		lineTuple = Bridson_Common.convertToTuple(line)
		originalSegmentTuple = Bridson_Common.convertToTuple(segment)
		# segmentTuple = Bridson_Common.convertToTuple(segment)
		for segmentTuple in cellContent0:
			# print("Checking ", segment, "against", segment0)
			segment0, currentLineTuple = segmentTuple

			# Only check for intersection if the segments are not from the same line.
			if lineTuple != currentLineTuple:
				intersection = Bridson_Common.line_intersect( originalSegmentTuple, Bridson_Common.convertToTuple(segment0) )
				if not intersection == None:
					# Indicates that there is an intersection.
					print("Intersection found at ", intersection)
					if intersection not in line:
						# Is point in the line?  If no, then it is a collision.
						collisionDetected = True
						pass
					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
						print("********* Intersection segment found in line", segment0, "************")
						# Is the segment in the line?  If it is, then probably not a collision because it is colinear.
						pass
					else:
						collisionDetected = True

		for segmentTuple in cellContent1:
			# print("Checking ", segment, "against", segment0)
			segment0, currentLineTuple = segmentTuple
			if lineTuple != currentLineTuple:
				intersection = Bridson_Common.line_intersect(originalSegmentTuple, Bridson_Common.convertToTuple(segment0))
				if not intersection == None:
					# Indicates that there is an intersection.
					print("Intersection found at", intersection)
					if intersection not in line:
						# Is the point in the line?  If no, then it is a collision.
						collisionDetected = True
						pass
					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
						print("********* Intersection segment found in line", segment0, "************")
						pass
					else:
						collisionDetected = True

		return collisionDetected




	def __addSegmentToCellMapping(self, segment, tupleLine):

		point1 = segment[0]
		point2 = segment[1]

		'''
		There are four cases to handle.
		Case A: Both lines belong to the same cell.  Prevent duplicates.
		Case B: Diagonal case.  Have to include the neighbouring cells
		The endpoints land in cells A and D.  We need to include cells B and C.
		
		Case C: Endpoints land in A and B.  No diagonal cells to include.
		Case D: Endpoints land in A and C.  No diagonal cells to include.
		_________
		| A | B |
		---------
		| C | D |
		---------
		'''

		cells = []
		firstCell = self.__findPointCell(point1)
		cells.append( firstCell )

		secondCell = self.__findPointCell(point2)
		if secondCell in cells:
			# Case A.
			pass
		elif self.__determineCaseB(firstCell, secondCell):
			cells.append( secondCell )
			cells.extend( self.__genDiagonalCells( firstCell, secondCell ))
		else:
			# Case C or Case D.  Simply add the second cell.
			cells.append( secondCell )

		# cells.append( self.__findPointCell(point2)  )
		# print("Adding segment:",self.convertToTuple(segment))
		self.segmentToCell[ Bridson_Common.convertToTuple( segment ) ] = cells
		self.__populateCellContent(segment, cells, tupleLine)

		# print("__addSegmentToCellMapping", segment, ". Belongs to cells", cells)
		pass



	def __genDiagonalCells(self, firstCell, secondCell):
		additionalCells = []
		x1,y1=firstCell
		x2,y2=secondCell
		additionalCells.append( (x2,y1) )
		additionalCells.append( (x1,y2) )
		return additionalCells

	def __determineCaseB(self, firstCell, secondCell):
		'''
		Determine if this is case B.
		:param firstCell:
		:param secondCell:
		:return:  True if Case B, else False.
		'''
		x1,y1 = firstCell
		x2,y2 = secondCell
		if x1 != x2 and y1 != y2:  # Condition for case 2.
			return True
		return False



	def __findPointCell(self, point):
		'''
		Based on the point, find the cell that contains the points.
		Return: coordinates of the cell that contains the point.
		'''
		x, y = point
		cellx = math.floor( x / self.cellsize )
		celly = math.floor( y / self.cellsize )

		# When pessimistic is set, we want the cells surrouding the current cell.
		# We can add the cells in the cardinal directions or all surrounding cells.

		return (cellx, celly)




	def findSegmentCells(self, segment):
		'''
		Find the cells that the segment belongs to.
		'''
		print("Segment belongs to cell(s)", self.segmentToCell[ Bridson_Common.convertToTuple(segment) ])
		return self.segmentToCell[ Bridson_Common.convertToTuple(segment) ]
		pass


	def __populateCellContent(self, segment, cells, tupleLine):
		for cell in cells:
			if cell in self.cellContent:
				self.cellContent[ cell ].append( (Bridson_Common.convertToTuple(segment), tupleLine) )
			else:
				cellContent = []
				cellContent.append( (Bridson_Common.convertToTuple(segment), tupleLine) )
				self.cellContent[ cell ] = cellContent


	def getCellContent(self, point):
		cellIndex = self.__findPointCell(point)
		if cellIndex not in self.cellContent:
			self.cellContent[cellIndex] = []
		print("Cell Content:", self.cellContent[ cellIndex ] )
		return self.cellContent[cellIndex]


	def cellNeighbourContent(self, coordinates):

		pass


	def neighbourSegments(self, coordinates):

		pass


	def findCollision(self, segment):

		pass


