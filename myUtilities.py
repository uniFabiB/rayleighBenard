import sys
import os
import datetime
import traceback
import warnings
import numpy
from shutil import copy

class utils:

	INFOFILENAME = "infoFile.txt"
	
	comm = None
	
	rank = 0
	
	outputFolder = ""
	
	infoFilePath = ""
	infoString = ""
	
	startTime = -1
	simulationId = -1
	simulationTime = 0
	
	recoveryScriptFileName = ""
	
	def print(self, *strings):
		if self.rank == 0:
			string = ""
			for stri in strings:
				string += str(stri) + " "
			print(datetime.datetime.now(), string)
			sys.stdout.flush()
			

	def __init__(self, comm = None, outputFolder = "output/"):
		if comm == None:
			self.rank = 0
		else:
			self.rank = comm.Get_rank()
			ensembleSize = comm.size
		self.comm = comm
		self.setStartTime(datetime.datetime.now())
		sys.excepthook = self.err
		self.print("starting")
		self.outputFolder = outputFolder
		self.setInfoFilePath(outputFolder+self.INFOFILENAME)
		self.putInfoInInfoString("startTime", self.startTime)
		self.generateSimulationId(self.startTime)
		self.checkIfFolderExists(outputFolder)
		self.checkIfFolderExists(outputFolder + "checkpoints/")
		self.deletePreviousInfoFileContent()
		self.putInfoInInfoString("outputFolder", outputFolder)
		self.putInfoInInfoString("ensembleSize", ensembleSize)
		
	#	sys.stdout.flush()
		comm.Barrier()
			
	
	def checkIfFolderExists(self, folder):
		if not os.path.isdir(folder):
			createFolderAnswer = self.askYesNoQuestion("directory ("+folder+") doesn't exist! create directory?", True)
			if createFolderAnswer:
				if self.rank == 0:
					os.mkdir(folder)
				self.print("creating output directory (",folder,")")
			else:
				raise Exception("output directory doesn't exist!")
			self.comm.Barrier()
	
	def setInfoFilePath(self, infoFilePath):
		self.infoFilePath = infoFilePath
		
	def deletePreviousInfoFileContent(self):
		# delete content of infofile
		if self.rank == 0:
			infoFile = open(self.infoFilePath,"w")
			infoFile.write("")
			infoFile.close()
		
	def generateSimulationId(self, startTime):
		simulationId = str(startTime) +str("_-_") +str(int(numpy.round(10000000000*numpy.random.rand())))
		simulationId = simulationId.replace(":","-")
		simulationId = simulationId.replace(" ","_")
		self.simulationId = simulationId
		self.putInfoInInfoString("simulationId", simulationId)
		return simulationId
	
	def generateRecoveryScript(self, script):
		if self.rank == 0:
			recoveryScriptFileName = "used_script_simulation_"+self.simulationId+".py"
			self.putInfoInInfoString("recoveryScriptFileName",recoveryScriptFileName)
			self.copyScriptToPath(script, self.outputFolder+recoveryScriptFileName)
		
		
	def copyScriptToPath(self, script, path):
		copy(os.path.realpath(script), path)
		
	
	def setStartTime(self, startTime):
		self.startTime = startTime
		

	def putInfoInInfoString(self, identifier, value):
		if self.rank == 0:
			self.infoString += self.addIdentifierAndValue(identifier, value)
	
	def addIdentifierAndValue(self, identifier, value):
		return "\n\t"+identifier+" = \t\t"+str(value)
	
	def setSimulationTime(self, simulationTime):
		self.simulationTime = simulationTime
		
	def err(self, type, value, tb):
		errorTime = datetime.datetime.now()
		if self.startTime == -1:
			completeDuration = "ka"
		else:
			completeDuration = errorTime - self.startTime
			
		print(errorTime,"Exception after",completeDuration)
		sys.stdout.flush()
		
		
		errorInfo = "\n\nERROR INFO"
		errorInfo += "\nscript stopped because of an error"
		errorInfo += self.addIdentifierAndValue("errorTime", errorTime)
		errorInfo += self.addIdentifierAndValue("errorSimulationTime",self.simulationTime)
		errorInfo += self.addIdentifierAndValue("errorCompleteDuration", completeDuration)
		
		errorInfo += self.addIdentifierAndValue("error type", type)
		errorInfo += self.addIdentifierAndValue("error value", value)
		errorInfo += self.addIdentifierAndValue("error traceback", traceback.format_exception(type, value, tb))
		
		infoFile = open(self.infoFilePath,"a")
		infoFile.write(errorInfo)
		infoFile.close()
		
		sys.__excepthook__(type, value, tb)
	
	def writeInfoFile(self):
		if self.rank == 0:
			infoFile = open(self.infoFilePath,"a")
			infoFile.write(self.infoString)
			infoFile.close()
			self.infoString = ""	
			
			
	def askYesNoQuestion(self, question, defaultAnswer):
		retVal = defaultAnswer
		if self.rank == 0:
			yes = {"yes","y", "ye", ""}
			no = {"no","n","\x1b"}   #\x1b = esc
			self.print(question, "(yes/enter or no)")
			answer = input()
			if answer in yes:
				retVal = True
			elif answer in no:
				retVal = False
			else:
				raise Exception("answer",answer, "is neither yes nor no")
		self.comm.Barrier()		## HAS TO BE RUN THROUGH BY EVERYONE
		return retVal
		
	def writeEndInfo(self):
		finishTime = datetime.datetime.now()
		completeDuration = finishTime - self.startTime
		self.putInfoInInfoString("FINISHED","")
		self.putInfoInInfoString("finishTime", finishTime)
		self.putInfoInInfoString("completeDuration", completeDuration)
		self.print("FINISHED")
		self.print("\tat\t", finishTime)
		self.print("\tafter\t\t    ", completeDuration)
		self.writeInfoFile()
			
	def warn(self, *warning):
		self.print("WARNING\n\t", *warning)
		cont = self.askYesNoQuestion("continue anyways?", True)
		if cont:
			pass
		else:
			raise Exception("didn't continue after warning")
		self.print("continuing")
		
