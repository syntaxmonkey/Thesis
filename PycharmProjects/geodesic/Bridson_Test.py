import subprocess

path = "../../boundary-first-flattening/build/"
# os.system(path + "bff-command-line " + path + "test1.obj " + path + "test1_out.obj --angle=1 --normalizeUVs ")
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
''''''

parameters =  " " + path + "test1.obj " + path + "test1_out.obj --flattenToDisk "

# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


#subprocess.call(path + 'bff-command-line' + parameters, shell=True)
subprocess.run(path + 'bff-command-line' + parameters, shell=True)

