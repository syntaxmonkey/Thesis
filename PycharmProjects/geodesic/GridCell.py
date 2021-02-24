import numpy as np
import math
import Bridson_Common

from numba import  jit_module

class LineEndPointIndex:
	def __init__(self):
		self.__lineEndPointIndex = {}
		pass


	def setLineIndeces(self, startIndex, endIndex, lineIndex):
		# lineTuple = Bridson_Common.convertToTuple(line)

		lineStartEndIndeces = [startIndex, endIndex]
		# self.__lineEndPointIndex[lineTuple] = lineStartEndIndeces
		self.__lineEndPointIndex[lineIndex] = lineStartEndIndeces
		# if lineTuple in self.__lineEndPointIndex:
		# 	lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		# else:
		# 	self.__lineEndPointIndex[ lineTuple ] = lineStartEndIndeces

		pass

	def updateLineStartIndex(self, line, startIndex, lineIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		# lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces = self.__lineEndPointIndex[lineIndex]

		if lineStartEndIndeces[0] > 0:
			lineStartEndIndeces[0] = startIndex
		pass


	def updateLineEndIndex(self, line, endIndex, lineIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		# lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces = self.__lineEndPointIndex[lineIndex]

		if lineStartEndIndeces[1] > 0:
			lineStartEndIndeces[1] = endIndex
		pass


	def updateLineCollisionStartIndex(self, line, lineIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		# lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces = self.__lineEndPointIndex[lineIndex]
		lineStartEndIndeces[0] = 0 - abs( lineStartEndIndeces[0] )


	def updateLineCollisionEndIndex(self, line, lineIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		# lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces = self.__lineEndPointIndex[lineIndex]
		lineStartEndIndeces[1] = 0 - abs( lineStartEndIndeces[1] )



	def getLineEndIndex(self, line, lineIndex):
		lineTuple = Bridson_Common.convertToTuple(line)
		# return self.__lineEndPointIndex[ lineTuple ]

		return self.__lineEndPointIndex[lineIndex]


	def isLineBounded(self, line, lineIndex):
		'''

		:param line:
		:return: True if both indeces are negative.
		'''
		lineTuple = Bridson_Common.convertToTuple(line)
		# lineStartEndIndeces = self.__lineEndPointIndex[ lineTuple ]
		lineStartEndIndeces = self.__lineEndPointIndex[lineIndex]
		if lineStartEndIndeces[0] < 0 and lineStartEndIndeces[1] < 0:
			return True
		return False


	def realignIndexes(self):
		# Iterate through all the indeces.  Convert any negative values to positive.
		for indeces in self.__lineEndPointIndex.values():
			indeces[0] = abs(indeces[0])
			indeces[1] = abs(indeces[1])


# class GridCells:
# 	def __init__(self, LimitX, LimitY, cellsize):
# 		self.LimitX = LimitX
# 		self.LimitY = LimitY
# 		self.cellsize = cellsize
#
# 		self.segmentToCell = {}
# 		self.cellContent = {} # Cell content is a list of tuples. The tuples are (segment, lineTuple).
# 		self.lineStartEndIndex = {}
#
# 		self.pessimisticCheck = True
# 		pass
#
#
# 	def addLine(self, line, startIndex, endIndex, lineIndex):
# 		## working for adding whole line.
# 		## working for partial lines.
# 		'''
# 		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
# 		Create an entry for (start,end) in segmentToCell.
# 		The cells will be based on start and end coordinates.
# 		'''
# 		tupleLine = Bridson_Common.convertToTuple(line)
# 		# Iterate through all the sub-segments.
# 		for index in range(startIndex, len(line) - endIndex - 1):
# 		# for index in range(len(line) - 1):
# 			segment = (line[index], line[index+1])
# 			self.__addSegmentToCellMapping(segment, tupleLine, lineIndex)
# 		# self.lineStartEndIndex[ Bridson_Common.convertToTuple( line ) ] = (startIndex, endIndex)
# 		pass
#
#
# 	def addSegmentBack(self, segments, line):
# 		'''
# 		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
# 		Create an entry for (start,end) in segmentToCell.
# 		The cells will be based on start and end coordinates.
# 		'''
# 		lineTuple = Bridson_Common.convertToTuple( line )
# 		# Iterate through all the sub-segments.
# 		for index in range( len(segments) - 1 ):
# 			segment = (segments[index], segments[index+1])
# 			print("addSegment:", segment)
# 			self.__addSegmentToCellMapping(segment, lineTuple)
# 		pass
#
# 	def addSegment(self, segments, line, lineIndex):
# 		'''
# 		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
# 		Create an entry for (start,end) in segmentToCell.
# 		The cells will be based on start and end coordinates.
# 		'''
# 		lineTuple = Bridson_Common.convertToTuple( line )
# 		# Iterate through all the sub-segments.
# 		for index in range( len(segments) - 1 ):
# 			segment = (segments[index], segments[index+1])
# 			print("addSegment:", segment)
# 			self.__addSegmentToCellMapping(segment, lineTuple, lineIndex)
# 		pass
#
#
#
# 	def detectCollisionBack(self, segment, line):
# 		'''
#
# 		:param segment:
# 		:param line:
# 		:return: True if there is a collision, False otherwise.
# 		'''
# 		'''
# 		Detect if there is a collision by adding this segment.
# 		'''
# 		print("Detecting Collision for segment:", Bridson_Common.convertToTuple( segment ) )
# 		# Find which cell it belongs to.
# 		cellContent0 = self.getCellContent( segment[0] )
# 		cellContent1 = self.getCellContent( segment[1] )
# 		# Check for collision with segments in the cell.
#
# 		collisionDetected = False
# 		lineTuple = Bridson_Common.convertToTuple(line)
# 		originalSegmentTuple = Bridson_Common.convertToTuple(segment)
# 		# segmentTuple = Bridson_Common.convertToTuple(segment)
# 		for segmentTuple in cellContent0:
# 			# print("Checking ", segment, "against", segment0)
# 			segment0, currentLineTuple = segmentTuple
#
# 			# Only check for intersection if the segments are not from the same line.
# 			if lineTuple != currentLineTuple:
# 				intersection = Bridson_Common.line_intersect( originalSegmentTuple, Bridson_Common.convertToTuple(segment0) )
# 				if not intersection == None:
# 					# Indicates that there is an intersection.
# 					print("Intersection found at ", intersection)
# 					if intersection not in line:
# 						# Is point in the line?  If no, then it is a collision.
# 						collisionDetected = True
# 						pass
# 					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
# 						print("********* Intersection segment found in line", segment0, "************")
# 						# Is the segment in the line?  If it is, then probably not a collision because it is colinear.
# 						pass
# 					else:
# 						collisionDetected = True
#
# 		for segmentTuple in cellContent1:
# 			# print("Checking ", segment, "against", segment0)
# 			segment0, currentLineTuple = segmentTuple
# 			if lineTuple != currentLineTuple:
# 				intersection = Bridson_Common.line_intersect(originalSegmentTuple, Bridson_Common.convertToTuple(segment0))
# 				if not intersection == None:
# 					# Indicates that there is an intersection.
# 					print("Intersection found at", intersection)
# 					if intersection not in line:
# 						# Is the point in the line?  If no, then it is a collision.
# 						collisionDetected = True
# 						pass
# 					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
# 						print("********* Intersection segment found in line", segment0, "************")
# 						pass
# 					else:
# 						collisionDetected = True
#
# 		return collisionDetected
#
#
# 	def detectCollision(self, segment, line, lineIndex):
# 		'''
# 		:param segment:
# 		:param line:
# 		:return: True if there is a collision, False otherwise.
# 		'''
# 		'''
# 		Detect if there is a collision by adding this segment.
# 		'''
# 		print("Detecting Collision for segment:", Bridson_Common.convertToTuple( segment ) )
# 		# Find which cell it belongs to.
# 		cellContent0 = self.getCellContent( segment[0] )
# 		cellContent1 = self.getCellContent( segment[1] )
# 		# Check for collision with segments in the cell.
#
# 		collisionDetected = False
# 		lineTuple = Bridson_Common.convertToTuple(line)
# 		originalSegmentTuple = Bridson_Common.convertToTuple(segment)
# 		# segmentTuple = Bridson_Common.convertToTuple(segment)
# 		for segmentTuple in cellContent0:
# 			# print("Checking ", segment, "against", segment0)
# 			segment0, currentLineIndex = segmentTuple
#
# 			# Only check for intersection if the segments are not from the same line.
# 			if lineIndex != currentLineIndex:
# 				intersection = Bridson_Common.line_intersect( originalSegmentTuple, Bridson_Common.convertToTuple(segment0) )
# 				if not intersection == None:
# 					# Indicates that there is an intersection.
# 					print("Intersection found at ", intersection)
# 					if intersection not in line:
# 						# Is point in the line?  If no, then it is a collision.
# 						collisionDetected = True
# 						pass
# 					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
# 						print("********* Intersection segment found in line", segment0, "************")
# 						# Is the segment in the line?  If it is, then probably not a collision because it is colinear.
# 						pass
# 					else:
# 						collisionDetected = True
#
# 		for segmentTuple in cellContent1:
# 			# print("Checking ", segment, "against", segment0)
# 			segment0, currentLineIndex = segmentTuple
# 			if lineIndex != currentLineIndex:
# 				intersection = Bridson_Common.line_intersect(originalSegmentTuple, Bridson_Common.convertToTuple(segment0))
# 				if not intersection == None:
# 					# Indicates that there is an intersection.
# 					print("Intersection found at", intersection)
# 					if intersection not in line:
# 						# Is the point in the line?  If no, then it is a collision.
# 						collisionDetected = True
# 						pass
# 					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
# 						print("********* Intersection segment found in line", segment0, "************")
# 						pass
# 					else:
# 						collisionDetected = True
#
# 		return collisionDetected
#
#
# 	def __addSegmentToCellMapping(self, segment, tupleLine, lineIndex):
#
# 		point1 = segment[0]
# 		point2 = segment[1]
#
# 		'''
# 		There are four cases to handle.
# 		Case A: Both lines belong to the same cell.  Prevent duplicates.
# 		Case B: Diagonal case.  Have to include the neighbouring cells
# 		The endpoints land in cells A and D.  We need to include cells B and C.
#
# 		Case C: Endpoints land in A and B.  No diagonal cells to include.
# 		Case D: Endpoints land in A and C.  No diagonal cells to include.
# 		_________
# 		| A | B |
# 		---------
# 		| C | D |
# 		---------
# 		'''
#
# 		cells = []
# 		firstCell = self.__findPointCell(point1)
# 		cells.append( firstCell )
#
# 		secondCell = self.__findPointCell(point2)
# 		if secondCell in cells:
# 			# Case A.
# 			pass
# 		elif self.__determineCaseB(firstCell, secondCell):
# 			# Case B.
# 			cells.append( secondCell )
# 			cells.extend( self.__genDiagonalCells( firstCell, secondCell ))
# 		else:
# 			# Case C or Case D.  Simply add the second cell.
# 			cells.append( secondCell )
#
# 		# cells.append( self.__findPointCell(point2)  )
# 		# print("Adding segment:",self.convertToTuple(segment))
# 		self.segmentToCell[ Bridson_Common.convertToTuple( segment ) ] = cells
# 		self.__populateCellContent(segment, cells, tupleLine, lineIndex)
#
# 		# print("__addSegmentToCellMapping", segment, ". Belongs to cells", cells)
# 		pass
#
#
#
# 	def __genDiagonalCells(self, firstCell, secondCell):
# 		additionalCells = []
# 		x1,y1=firstCell
# 		x2,y2=secondCell
# 		additionalCells.append( (x2,y1) )
# 		additionalCells.append( (x1,y2) )
# 		return additionalCells
#
# 	def __determineCaseB(self, firstCell, secondCell):
# 		'''
# 		Determine if this is case B.
# 		:param firstCell:
# 		:param secondCell:
# 		:return:  True if Case B, else False.
# 		'''
# 		x1,y1 = firstCell
# 		x2,y2 = secondCell
# 		if x1 != x2 and y1 != y2:  # Condition for case 2.
# 			return True
# 		return False
#
#
#
# 	def __findPointCell(self, point):
# 		'''
# 		Based on the point, find the cell that contains the points.
# 		Return: coordinates of the cell that contains the point.
# 		'''
# 		x, y = point
# 		cellx = math.floor( x / self.cellsize )
# 		celly = math.floor( y / self.cellsize )
#
# 		# When pessimistic is set, we want the cells surrouding the current cell.
# 		# We can add the cells in the cardinal directions or all surrounding cells.
#
# 		return (cellx, celly)
#
#
#
#
# 	def findSegmentCells(self, segment):
# 		'''
# 		Find the cells that the segment belongs to.
# 		'''
# 		print("Segment belongs to cell(s)", self.segmentToCell[ Bridson_Common.convertToTuple(segment) ])
# 		return self.segmentToCell[ Bridson_Common.convertToTuple(segment) ]
# 		pass
#
#
# 	def __populateCellContent(self, segment, cells, tupleLine, lineIndex):
# 		for cell in cells:
# 			if cell in self.cellContent:
# 				# self.cellContent[ cell ].append( (Bridson_Common.convertToTuple(segment), tupleLine) )
# 				self.cellContent[cell].append((Bridson_Common.convertToTuple(segment), lineIndex))
# 			else:
# 				cellContent = []
# 				# cellContent.append( (Bridson_Common.convertToTuple(segment), tupleLine) )
# 				cellContent.append((Bridson_Common.convertToTuple(segment), lineIndex))
# 				self.cellContent[ cell ] = cellContent
#
#
# 	def getCellContent(self, point):
# 		cellIndex = self.__findPointCell(point)
# 		if cellIndex not in self.cellContent:
# 			self.cellContent[cellIndex] = []
# 		print("Cell Content:", self.cellContent[ cellIndex ] )
# 		return self.cellContent[cellIndex]
#
#
# 	def cellNeighbourContent(self, coordinates):
#
# 		pass
#
#
# 	def neighbourSegments(self, coordinates):
#
# 		pass
#
#
# 	def findCollision(self, segment):
#
# 		pass


class GridCells2:
	def __init__(self, LimitX, LimitY, cellsize):
		self.LimitX = LimitX
		self.LimitY = LimitY
		self.adjustLimits()

		self.cellsize = cellsize

		self.segmentToCell = {}
		self.cellContent = {}  # Cell content is a list of tuples. The tuples are (segment, lineTuple).
		self.lineStartEndIndex = {}
		self.segmentMapLineIndex = {} # Will map the segment to a lineIndex.

		self.pessimisticCheck = True
		pass

	def adjustLimits(self):
		# Sort the limits such that X1 is smaller than X2.
		if self.LimitX[0] > self.LimitX[1]:
			self.LimitX = (self.LimitX[1], self.LimitX[0])
		if self.LimitY[0] > self.LimitY[1]:
			self.LimitY = (self.LimitY[1], self.LimitY[0])


	def addLine(self, line, startIndex, endIndex, lineIndex):
		## working for adding whole line.
		## working for partial lines.
		'''
		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
		Create an entry for (start,end) in segmentToCell.
		The cells will be based on start and end coordinates.
		'''
		# tupleLine = Bridson_Common.convertToTuple(line)
		# Iterate through all the sub-segments.
		# print("StartIndex:endIndex:", startIndex, endIndex)
		# print("Line:", line)
		for index in range(startIndex, endIndex):
			# for index in range(len(line) - 1):
			segment = (line[index], line[index + 1])
			self.__addSegmentToCellMapping(segment, lineIndex)
		# self.lineStartEndIndex[ Bridson_Common.convertToTuple( line ) ] = (startIndex, endIndex)
		pass

	def addSegmentBackup(self, segments, line):
		'''
		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
		Create an entry for (start,end) in segmentToCell.
		The cells will be based on start and end coordinates.
		'''
		lineTuple = Bridson_Common.convertToTuple(line)
		# Iterate through all the sub-segments.
		for index in range(len(segments) - 1):
			segment = (segments[index], segments[index + 1])
			# print("addSegment:", segment)
			self.__addSegmentToCellMapping(segment, lineTuple)
		pass

	def addSegment(self, segments, lineIndex):
		'''
		When adding a new segment, we need to iterate through all the sub-segments: (start,end)
		Create an entry for (start,end) in segmentToCell.
		The cells will be based on start and end coordinates.
		'''
		# lineTuple = Bridson_Common.convertToTuple(line)
		# Iterate through all the sub-segments.
		for index in range(len(segments) - 1):
			segment = (segments[index], segments[index + 1])
			# print("addSegment:", segment)
			self.__addSegmentToCellMapping(segment, lineIndex)
			self.segmentMapLineIndex[ Bridson_Common.convertToTuple(segment) ] = lineIndex
		pass

	def detectCollisionBackup(self, segment, line):
		'''

		:param segment:
		:param line:
		:return: True if there is a collision, False otherwise.
		'''
		'''
		Detect if there is a collision by adding this segment.
		'''
		# print("Detecting Collision for segment:", Bridson_Common.convertToTuple(segment))
		# Find which cell it belongs to.
		cellContent0 = self.getCellContent(segment[0])
		cellContent1 = self.getCellContent(segment[1])
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
				intersection = Bridson_Common.line_intersect(originalSegmentTuple, Bridson_Common.convertToTuple(segment0))
				if not intersection == None:
					# Indicates that there is an intersection.
					# print("Intersection found at ", intersection)
					if intersection not in line:
						# Is point in the line?  If no, then it is a collision.
						collisionDetected = True
						pass
					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
						# print("********* Intersection segment found in line", segment0, "************")
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
					# print("Intersection found at", intersection)
					if intersection not in line:
						# Is the point in the line?  If no, then it is a collision.
						collisionDetected = True
						pass
					elif Bridson_Common.findSegment(originalSegmentTuple, lineTuple):
						# print("********* Intersection segment found in line", segment0, "************")
						pass
					else:
						collisionDetected = True

		return collisionDetected

	def detectCollision(self, segment, lineIndex):
		'''
		:param segment:
		:param line:
		:return: True if there is a collision, False otherwise.
		'''
		'''
		Detect if there is a collision by adding this segment.
		'''
		# print("original Segment Tuple:", segment)
		originalSegmentTuple = Bridson_Common.convertToTuple(segment)
		# print("original Segment Tuple:", originalSegmentTuple)
		# Consider the case where the segment has fallen outside the limits.
		if not( originalSegmentTuple[0][0] > self.LimitX[0] and originalSegmentTuple[0][0] < self.LimitX[1] ):
			return True
		if not( originalSegmentTuple[1][0] > self.LimitX[0] and originalSegmentTuple[1][0] < self.LimitX[1] ):
			return True

		if not( originalSegmentTuple[0][1] > self.LimitY[0] and originalSegmentTuple[0][1] < self.LimitY[1] ):
			return True
		if not( originalSegmentTuple[1][1] > self.LimitY[0] and originalSegmentTuple[1][1] < self.LimitY[1] ):
			return True

		# print("Detecting Collision for segment:", originalSegmentTuple)
		# Find which cell it belongs to.
		cellContent0 = self.getCellContent(segment[0])
		cellContent1 = self.getCellContent(segment[1])
		# Check for collision with segments in the cell.

		collisionDetected = False
		# lineTuple = Bridson_Common.convertToTuple(line)

		# segmentTuple = Bridson_Common.convertToTuple(segment)
		for segmentTuple in cellContent0:
			# print("Checking ", segment, "against", segment0)
			segment0, currentLineIndex = segmentTuple

			# Only check for intersection if the segments are not from the same line.
			if lineIndex != currentLineIndex:
				intersection = Bridson_Common.line_intersect(originalSegmentTuple, Bridson_Common.convertToTuple(segment0))
				if not intersection == None:
					# Indicates that there is an intersection.
					# print("Intersection found at ", intersection)
					# Does the segment belong to the the current line index?
					if Bridson_Common.convertToTuple(segment0) in self.segmentMapLineIndex and self.segmentMapLineIndex[ Bridson_Common.convertToTuple(segment0) ] == lineIndex:
						# Other segment does belong to current line, therefore, not a real collision.
						pass
					else:
						collisionDetected = True

		for segmentTuple in cellContent1:
			# print("Checking ", segment, "against", segment0)
			segment0, currentLineIndex = segmentTuple
			if lineIndex != currentLineIndex:
				intersection = Bridson_Common.line_intersect(originalSegmentTuple, Bridson_Common.convertToTuple(segment0))
				if not intersection == None:
					# Indicates that there is an intersection.
					# print("Intersection found at", intersection)
					# Does the segment belong to the the current line index?
					if Bridson_Common.convertToTuple(segment0) in self.segmentMapLineIndex and self.segmentMapLineIndex[Bridson_Common.convertToTuple(segment0)] == lineIndex:
						# Other segment does belong to current line, therefore, not a real collision.
						pass
					else:
						collisionDetected = True

		return collisionDetected



	def __addSegmentToCellMapping(self, segment, lineIndex):

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
		cells.append(firstCell)

		secondCell = self.__findPointCell(point2)
		if secondCell in cells:
			# Case A.
			pass
		elif self.__determineCaseB(firstCell, secondCell):
			# Case B.
			cells.append(secondCell)
			cells.extend(self.__genDiagonalCells(firstCell, secondCell))
		else:
			# Case C or Case D.  Simply add the second cell.
			cells.append(secondCell)

		# cells.append( self.__findPointCell(point2)  )
		# print("Adding segment:",self.convertToTuple(segment))
		self.segmentToCell[Bridson_Common.convertToTuple(segment)] = cells
		self.__populateCellContent(segment, cells, lineIndex)

		# print("__addSegmentToCellMapping", segment, ". Belongs to cells", cells)
		pass

	def __genDiagonalCells(self, firstCell, secondCell):
		additionalCells = []
		x1, y1 = firstCell
		x2, y2 = secondCell
		additionalCells.append((x2, y1))
		additionalCells.append((x1, y2))
		return additionalCells

	def __determineCaseB(self, firstCell, secondCell):
		'''
		Determine if this is case B.
		:param firstCell:
		:param secondCell:
		:return:  True if Case B, else False.
		'''
		x1, y1 = firstCell
		x2, y2 = secondCell
		if x1 != x2 and y1 != y2:  # Condition for case 2.
			return True
		return False

	def __findPointCell(self, point):
		'''
		Based on the point, find the cell that contains the points.
		Return: coordinates of the cell that contains the point.
		'''
		x, y = point
		cellx = math.floor(x / self.cellsize)
		celly = math.floor(y / self.cellsize)

		# When pessimistic is set, we want the cells surrouding the current cell.
		# We can add the cells in the cardinal directions or all surrounding cells.

		return (cellx, celly)

	def findSegmentCells(self, segment):
		'''
		Find the cells that the segment belongs to.
		'''
		# print("Segment belongs to cell(s)", self.segmentToCell[Bridson_Common.convertToTuple(segment)])
		return self.segmentToCell[Bridson_Common.convertToTuple(segment)]
		pass

	def __populateCellContent(self, segment, cells, lineIndex):
		for cell in cells:
			if cell in self.cellContent:
				# self.cellContent[ cell ].append( (Bridson_Common.convertToTuple(segment), tupleLine) )
				self.cellContent[cell].append((Bridson_Common.convertToTuple(segment), lineIndex))
			else:
				cellContent = []
				# cellContent.append( (Bridson_Common.convertToTuple(segment), tupleLine) )
				cellContent.append((Bridson_Common.convertToTuple(segment), lineIndex))
				self.cellContent[cell] = cellContent

	def getCellContent(self, point):
		cellIndex = self.__findPointCell(point)
		if cellIndex not in self.cellContent:
			self.cellContent[cellIndex] = []
		# print("Cell Content:", self.cellContent[cellIndex])
		return self.cellContent[cellIndex]

	def cellNeighbourContent(self, coordinates):

		pass

	def neighbourSegments(self, coordinates):

		pass

	def findCollision(self, segment):

		pass

class LineObject:
	def __init__(self, linePoints, lineIndex, regionIndex, erosionIndex=Bridson_Common.erosionCount):
		# print("Creating LineObject:", linePoints)
		self.__linePoints = linePoints # Should be private
		self.lineIndex = lineIndex  # Publicly accessible.
		self.unbound()
		self.endIndeces = [0, 0] # Line endpoint indeces.
		self.calculateStartIndeces( erosionIndex)
		self.regionIndex = regionIndex
		self.pointIntensity = [] # Will store the intensities for each point.
		self.intensityAverage = 0 # Will be the intensity average for the line.
		self.calculateDirection()

	def __eq__(self, other):
		if self.lineIndex == other.lineIndex:
			return True
		else:
			return False

	def unbound(self):
		# Indicates if either end is bounded.  Index 0 is start index.  Index 1 is end index.
		self.endBounded = [False, False]


	def wiggleLineBak(self):
		pointWiggles = np.zeros( (len(self.__linePoints), 2) ) # Create wiggle array the same size as self.__linePoints.
		for index in range(self.endIndeces[0]+1, self.endIndeces[1]-1):
			currentIntensity = self.pointIntensity[ index ]
			currentPoint = self.__linePoints[index]
			previousPoint = self.__linePoints[index - 1]
			intensityDiff = currentIntensity - self.intensityAverage
			# Calculate the vector for the point.
			pointVector = (currentPoint[0] - previousPoint[0], currentPoint[1] - previousPoint[1])
			hypotenuse = math.sqrt(pointVector[0]*pointVector[0] + pointVector[1]*pointVector[1])

			if pointVector == 0:
				hypotenuse = pointVector[0] if pointVector[0] > pointVector[1] else pointVector[1]

			# Rotate the vector 90 degrees CW to get normal vector.
			# Make the longest component of the normal vector equal to 1.
			scalingFactor = 1 / hypotenuse * Bridson_Common.wiggleScale * abs(intensityDiff) / 255
			if intensityDiff > 0:
				normalVector = (pointVector[1] * scalingFactor, -pointVector[0] * scalingFactor)
			else:
				normalVector = ( -pointVector[1] * scalingFactor, pointVector[0] * scalingFactor)

			pointWiggles[ index ]  = normalVector

		self.__linePoints = self.__linePoints + pointWiggles


	def wiggleLine(self, smallBlurArray, largeBlurArray, flip):
		pointWiggles = np.zeros( (len(self.__linePoints), 2) ) # Create wiggle array the same size as self.__linePoints.
		for index in range(self.endIndeces[0]+1, self.endIndeces[1]-1):
			point = self.__linePoints[ index ]
			x, y = point
			x = int(x)
			y = int(y)
			# pointIntensity = smallBlurArray[y * flip, x]
			# averageIntensity = largeBlurArray[y * flip, x]

			# currentIntensity = self.pointIntensity[ index ]
			currentPoint = self.__linePoints[index]
			previousPoint = self.__linePoints[index - 1]
			# intensityDiff = pointIntensity - averageIntensity
			# intensityDiff = int(smallBlurArray[y * flip, x]) - int(largeBlurArray[y * flip, x])
			intensityDiff = int(smallBlurArray[y * flip, x]) - int(largeBlurArray[y * flip, x])

			# Calculate the vector for the point.
			pointVector = (currentPoint[0] - previousPoint[0], currentPoint[1] - previousPoint[1])
			hypotenuse = math.sqrt(pointVector[0]*pointVector[0] + pointVector[1]*pointVector[1])

			if pointVector == 0:
				hypotenuse = pointVector[0] if pointVector[0] > pointVector[1] else pointVector[1]

			# Rotate the vector 90 degrees CW to get normal vector.
			# Make the longest component of the normal vector equal to 1.
			scalingFactor = 1 / hypotenuse * Bridson_Common.wiggleScale * abs(intensityDiff) / 255
			if scalingFactor == 0:
				scalingFactor = 1

			if intensityDiff > 0:
				normalVector = (pointVector[1] * scalingFactor, -pointVector[0] * scalingFactor)
			else:
				normalVector = ( -pointVector[1] * scalingFactor, pointVector[0] * scalingFactor)

			pointWiggles[ index ]  = normalVector

		self.__linePoints = self.__linePoints + pointWiggles


	def populateIntensity(self, greyImageArray, smallBlurArray, largeBlurArray, flip):
		self.pointIntensity = [0] * len(self.__linePoints)

		intensityTotal = 0
		count = 0
		for index in range( self.endIndeces[0], self.endIndeces[1], 1):
			point = self.__linePoints[ index ]
			x, y = point
			x = int(x)
			y = int(y)
			intensity = smallBlurArray[y * flip,x] # Remember the Image array is Numpy array.  The x,y are reversed.
			intensityTotal += intensity
			self.pointIntensity[ index ] = smallBlurArray[y*flip, x] # Get the point intensity from the small blur array.
			count += 1

		if count > 0:
			self.intensityAverage = intensityTotal / count
		# print("Line Intensity:", self.pointIntensity)
		# print("Line Intensity average:", self.intensityAverage)

	def calculateDirection(self):
		# Determine the direction vector of the line.
		# [Last element] - [first element]
		lastPoint = self.__linePoints[-1]
		firstPoint = self.__linePoints[0]
		self.directionVector = ( lastPoint[0] - firstPoint[0], lastPoint[1] - firstPoint[1] )

	def getDirectionVector(self):
		return self.directionVector


	def alignDirection(self, baseVector):
		dotProduct = np.dot( baseVector, self.directionVector )
		# print("baseVector:", baseVector, "self.directionVector:", self.directionVector)
		if dotProduct > 0:
			# The direction of the vectors are the same.  Change nothing.
			pass
		else:
			# print("DotProduct:", dotProduct, "Aligning line:", self.lineIndex)
			# Change the direction of the points in the line.
			self.__linePoints = np.flip( self.__linePoints, axis=0 )
			lineLength = len(self.__linePoints)
			self.endBounded = [ self.endBounded[1], self.endBounded[0] ]
			self.endIndeces = [ lineLength - self.endIndeces[1], lineLength - self.endIndeces[0] ]
			self.calculateDirection()







	def extendLineDetectCollision(self, allGridCells):
		segmentAdded = False
		if self.endBounded[0] == False:
			self.extendStartDirection()
			if allGridCells.detectCollision(self.getNextStartSegment(),  self.lineIndex):
				self.endBounded[0] = True
			else:
				#
				allGridCells.addSegment(self.getNextStartSegment(), self.lineIndex)
				self.endIndeces[0] -= 1
				segmentAdded = True


		if True:
			if self.endBounded[1] == False:
				self.extendEndDirection()
				# if self.endIndeces[1] <= len(self.__linePoints) - 2:  # the index is less than or equal to because we are using the index for slicing.
				if allGridCells.detectCollision( self.getNextEndSegment(),  self.lineIndex ):
					# Collision detected if we extend the start direction.  Set collision detection flag.
					self.endBounded[1] = True
				else:
					# No Collision detected.  Add segment to allGridCells.
					allGridCells.addSegment( self.getNextEndSegment(),  self.lineIndex)
					self.endIndeces[1] += 1
					segmentAdded = True

		return segmentAdded



	def detectCollisionAdjustIndeces(self, allGridCells):
		'''
		This method will check all segments for collision.
		If there are no collisions detected, then add the line.
		Otherwise, do not add line.
		'''
		self.setIndecesAggressiveErosion()
		collisionDetected = False
		return collisionDetected

		self.highlightStartingIndex = None
		# Iterate through the segments until there is a collision.
		for index in range(self.endIndeces[0], self.endIndeces[1]-1):
			if allGridCells.detectCollision( self.__linePoints[index:index+2],  self.lineIndex):
				collisionDetected = True
				self.originalIndeces = self.endIndeces
				# self.endIndeces = [0, index] # Set the line index to what currently fits.
				if Bridson_Common.randomLineSeed:
					self.setRandomEndIndeces()
				else:
					# self.setIndecesMiddlePoint() # We have detected a collision.
					self.setIndecesAggressiveErosion()
					# self.setIndecesStart()
				break # Stop the iteration because we have detected a collision.
		# Add the line to the gridcell.
		return collisionDetected

	def setRandomEndIndeces(self):
		self.originalIndeces = self.endIndeces
		lineLength = len(self.__linePoints)
		midPoint = np.random.randint(1, lineLength-2)
		self.endIndeces = [midPoint, midPoint + 1]
		self.highlightStartingIndex = self.endIndeces.copy()

	def setIndecesMiddlePoint(self):
		self.originalIndeces = self.endIndeces
		lineLength = len(self.__linePoints)
		midPoint = math.floor(lineLength / 2)
		self.endIndeces = [midPoint, midPoint + 1]
		self.highlightStartingIndex = self.endIndeces.copy()


	def setIndecesAggressiveErosion(self):
		aggressiveScale = Bridson_Common.aggressiveScale
		# print("Calling aggressive erosion")
		self.originalIndeces = self.endIndeces
		lineLength = len(self.__linePoints)
		if lineLength > Bridson_Common.erosionCount * 2 * aggressiveScale + 1:
			erosionIndex = Bridson_Common.erosionCount * aggressiveScale
			self.endIndeces[0] = erosionIndex
			self.endIndeces[1] = lineLength - erosionIndex
		else:
			self.setIndecesMiddlePoint()
		self.highlightStartingIndex = self.endIndeces.copy()


	def setIndecesStart(self):
		self.endIndeces = [0, 1]


	def calculateStartIndeces(self, erosionIndex):
		# print("calculateStartIndeces erosionIndex:", erosionIndex)
		lineLength = len(self.__linePoints)
		if erosionIndex == -1:
			self.endIndeces = [0, lineLength-1]
		elif erosionIndex == 0 or lineLength < erosionIndex * 2:
			midPoint = math.floor( lineLength / 2 )
			self.endIndeces = [midPoint, midPoint + 1]
		else:
			self.endIndeces = [erosionIndex, lineLength - erosionIndex]
		return



	def extendStartDirection(self):
		# Add a segment to the beginning of the line.
		point1 = self.__linePoints[1]
		point0 = self.__linePoints[0]

		deltax = point1[0] - point0[0]
		deltay = point1[1] - point0[1]

		divisor = math.sqrt(deltax * deltax + deltay * deltay)
		# if abs(deltax) > abs(deltay):
		# 	divisor = abs(deltax)
		# else:
		# 	divisor = abs(deltay)

		if divisor == 0:
			divisor = deltax if deltax > deltay else deltay
		deltax = deltax / divisor
		deltay = deltay / divisor

		newPoint = [point0[0] - deltax, point0[1] - deltay]
		# print("Before:",self.__linePoints)
		# print("newPoint:", newPoint)
		self.__linePoints = np.insert(self.__linePoints, 0, newPoint, axis=0)
		# print("After:",self.__linePoints)
		self.endIndeces[0] += 1
		self.endIndeces[1] += 1
		# print("extendStartDirection - new point:", newPoint, "existing line:", self.__linePoints)
		pass




	def extendEndDirection(self):
		# Add a segment to the end of the line.
		lastPoint = self.__linePoints[-1]
		secondLastPoint = self.__linePoints[-2]

		deltax = secondLastPoint[0] - lastPoint[0]
		deltay = secondLastPoint[1] - lastPoint[1]
		divisor = math.sqrt(deltax * deltax + deltay * deltay)

		# if abs(deltax) > abs(deltay):
		# 	divisor = abs(deltax)
		# else:
		# 	divisor = abs(deltay)
		if divisor == 0:
			divisor = deltax if deltax > deltay else deltay
		deltax = deltax / divisor
		deltay = deltay / divisor

		newPoint = (lastPoint[0] - deltax, lastPoint[1] - deltay)

		self.__linePoints = np.insert(self.__linePoints, len(self.__linePoints), newPoint, axis=0)
		# print("extendEndDirection - new point:", newPoint, "existing line:", self.__linePoints)
		pass


	def growDetectCollision(self, allGridCells):
		segmentAdded = False
		# Extend the starting side.
		if self.endBounded[0] == False:
			# print("Start Index:", self.endIndeces[0])
			if self.endIndeces[0] >= 1:  # Index needs to be > 1 because segments spans 3 indeces.
				if allGridCells.detectCollision( self.getNextStartSegment(), self.lineIndex ):
					# Collision detected if we extend the start direction.  Set collision detection flag.
					self.endBounded[0] = True
					# print("Start Index bounded:", self.endIndeces[0])
				else:
					# No Collision detected.  Add segment to allGridCells.
					allGridCells.addSegment( self.getNextStartSegment(),  self.lineIndex)
					self.endIndeces[0] -= 1
					segmentAdded = True
			else:
				self.endBounded[0] = True
				# print("Start Index bounded:", self.endIndeces[0])

		# Extend the ending side.
		if self.endBounded[1] == False:
			# print("End Index:", self.endIndeces[1])
			if self.endIndeces[1] <= len(self.__linePoints) - 2:  # the index is less than or equal to because we are using the index for slicing.
				if allGridCells.detectCollision( self.getNextEndSegment(), self.lineIndex ):
					# Collision detected if we extend the start direction.  Set collision detection flag.
					self.endBounded[1] = True
					# print("End Index bounded:", self.endIndeces[1], len(self.__linePoints))
				else:
					# No Collision detected.  Add segment to allGridCells.
					allGridCells.addSegment( self.getNextEndSegment(),  self.lineIndex)
					self.endIndeces[1] += 1
					segmentAdded = True
			else:
				self.endBounded[1] = True
				# print("End Index bounded:", self.endIndeces[1], len(self.__linePoints))
		return segmentAdded



	def getLinePoints(self):
		return self.__linePoints

	def getBoundedLinePoints(self):
		'''
		:return: Return the line points based on the current endpoint indeces.
		'''
		# print("getBoundedLinePoints indeces:", self.endIndeces, "for line length:", len(self.__linePoints))
		return self.__linePoints[ self.endIndeces[0] : self.endIndeces[1] + 1 ]  # Need to increment the end index by 1 because the array indexing is not inclusive of last element.

	def getStartIndex(self):
		return self.endIndeces[0]

	def getEndIndex(self):
		return self.endIndeces[1]

	def getLineIndex(self):
		return self.lineIndex

	def getNextStartSegment(self):
		'''
		If line is not bounded on the start index side, return the next segment.
		If it is bounded, return None.
		'''
		if self.endBounded[0] == False:
			# print("Returning value for self.endIndeces[0]", self.endIndeces[0] - 1)
			# print("from Line:", self.__linePoints)
			return self.__linePoints[ self.endIndeces[0]-1: self.endIndeces[0]+1 ]
		else:
			return None


	def getNextEndSegment(self):
		'''
		If line is not bounded on the end index side, return the next segment.
		If it is bounded, return None.
		'''
		if self.endBounded[1] == False:
			# Handle the normal case.
			return self.__linePoints[ self.endIndeces[1]: self.endIndeces[1] + 2 ]
		else:
			return None




	def isBounded(self):
		'''

		:return: True if both sides of the line are bounded.
		'''
		if (self.endBounded[0] == True) and (self.endBounded[1] == True):
			# print(">>>>>>>>>>>> Returning true for isBounded: ", self.lineIndex)
			return True
		else:
			# print("***** Returning false for isBounded: ", self.lineIndex, "*****")
			return False


jit_module(nopython=True)