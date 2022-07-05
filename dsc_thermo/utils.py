import os
'''
splits a dsc output file into two output files
    readfile_name: name of the file to be split
    writefile1_name: name of the file that will contain the first part of the data
    writefile2_name: name of the file that will contain the second part of the data
    step: step at which to partition the data
    repeat_step (T/F): if True, both files will store the data from the partition step
        if False, only the second file will store the data from this step
'''
def split_file(readfile_name, writefile1_name, writefile2_name, step, repeat_step=True):
    if os.path.isfile(writefile1_name):
        raise RuntimeError("{} already exists".format(writefile1_name))
    if os.path.isfile(writefile2_name):
        raise RuntimeError("{} already exists".format(writefile2_name))
        
    reading_header = True
    writing_file1 = True
    writing_file2 = False
    header_list = []
    with open(readfile_name, "r", encoding="Windows-1252") as readfile:
        with open(writefile1_name, "w", encoding="Windows-1252") as writefile1:
            writefile1.write("#split from {}\n\n".format(readfile_name))
            with open(writefile2_name, "w", encoding="Windows-1252") as writefile2:
                writefile2.write("#split from {}\n\n".format(readfile_name))
                line = readfile.readline()
                while line:
                    if reading_header:
                        header_list.append(line)
                        if "\t          \t" in line:
                            reading_header = False
                            
                    if line.startswith("{}) DSC 8500".format(step)):
                        if not repeat_step:
                            writing_file1 = False
                        writing_file2 = True
                        writefile2.write("".join(header_list))
                    elif repeat_step and line.startswith("{}) DSC 8500".format(step+1)):
                        writing_file1 = False
                        
                    if writing_file1:  
                        writefile1.write(line)
                    if writing_file2 and not line.startswith("{}) DSC 8500".format(step)):
                        writefile2.write(line)
                    line = readfile.readline()
