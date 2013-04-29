__author__ = 'behzadbehzadan'
import unicodecsv
import pickle
import csv

def write_csv_file(file_name, tab_delimiter_flag, unicode_flag, arg):
    print 'writing file:', file_name
    f_w = open(file_name + '.csv', 'w')
    if unicode_flag:
        if tab_delimiter_flag:
            writer = unicodecsv.writer(f_w, 'excel-tab', encoding='utf-8')
        else:
            writer = unicodecsv.writer(f_w, encoding='utf-8')
    else:
        if tab_delimiter_flag:
            writer = csv.writer(f_w, 'excel-tab')
        else:
            writer = csv.writer(f_w)
    writer.writerows(arg)
    f_w.close()

def read_csv_file(file_name, tab_delimiter_flag, unicode_flag):
    print 'reading file:', file_name
    contents = []
    f_r = open(file_name + '.csv', 'r')
    if unicode_flag:
        if tab_delimiter_flag:
            reader = unicodecsv.reader(f_r, 'excel-tab', encoding='utf-8')
        else:
            reader = unicodecsv.reader(f_r, encoding='utf-8')
    else:
        reader = csv.reader(f_r)
    for row in reader:
        contents.append(row)
    f_r.close()
    return contents

def pickle_to_file(file_name, arg):
    f_w = open(file_name, 'w')
    pickle.dump(arg, f_w)
    f_w.close()

def read_pickled_file(file_name):
    f_r = open(file_name, 'r')
    contents = pickle.load(f_r)
    f_r.close()
    return contents

