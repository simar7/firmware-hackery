#!/usr/bin/env python
import sys
from os import listdir, path
from distutils.core import setup

WIDTH = 115

# Check for pre-requisite modules
print "checking pre-requisites"
try:
	import magic
	try:
		magic.MAGIC_NO_CHECK_TEXT
	except Exception, e:
		print "\n", "*" * WIDTH
		print "Pre-requisite failure:", str(e)
		print "It looks like you have an old or incompatible magic module installed."
		print "Please install the official python-magic module, or download and install it from source: ftp://ftp.astron.com/pub/file/"
		print "*" * WIDTH, "\n"
		sys.exit(1)
except Exception, e:
	print "\n", "*" * WIDTH
	print "Pre-requisite failure:", str(e)
	print "Please install the python-magic module, or download and install it from source: ftp://ftp.astron.com/pub/file/"
	print "*" * WIDTH, "\n"
	sys.exit(1)

try:
	import matplotlib
	matplotlib.use('Agg')
	import matplotlib.pyplot
	import numpy
except Exception, e:
	print "\n", "*" * WIDTH
	print "Pre-requisite check warning:", str(e)
	print "To take advantage of this tool's entropy plotting capabilities, please install the python-matplotlib module."
	print "*" * WIDTH, "\n"
	
	if raw_input('Continue installation without this module (Y/n)? ').lower().startswith('n'):
		print 'Quitting...\n'
		sys.exit(1)
		

# Generate a new magic file from the files in the magic directory
print "generating binwalk magic file"
magic_files = listdir("magic")
magic_files.sort()
fd = open("binwalk/magic/binwalk", "wb")
for magic in magic_files:
	fpath = path.join("magic", magic)
	if path.isfile(fpath):
		fd.write(open(fpath).read())
fd.close()

# The data files to install along with the binwalk module
install_data_files = ["magic/*", "config/*", "plugins/*"]

# Install the binwalk module, script and support files
setup(	name = "binwalk",
	version = "1.2.2",
	description = "Firmware analysis tool",
	author = "Craig Heffner",
	url = "http://binwalk.googlecode.com",

	requires = ["magic", "matplotlib.pyplot"],	
	packages = ["binwalk"],
	package_data = {"binwalk" : install_data_files},
	scripts = ["bin/binwalk"],
)
