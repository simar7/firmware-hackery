import zlib
import math
import os.path
import plugins
import common

class FileEntropy(object):
	'''
	Class for analyzing and plotting data entropy for a file.
	Preferred to use the Entropy class instead of calling FileEntropy directly.
	'''

	DEFAULT_BLOCK_SIZE = 1024
	ENTROPY_TRIGGER = 0.9
	ENTROPY_MAX = 0.95
	FILE_FORMAT = 'svg'

	def __init__(self, file_name=None, fd=None, binwalk=None, offset=0, length=None, block=DEFAULT_BLOCK_SIZE, plugins=None):
		'''
		Class constructor.

		@file_name - The path to the file to analyze.
		@fd        - A file object to analyze data from.
		@binwalk   - An instance of the Binwalk class.
		@offset    - The offset into the data to begin analysis.
		@length    - The number of bytes to analyze.
		@block     - The size of the data blocks to analyze.
		@plugins   - Instance of the Plugins class.

		Returns None.
		'''
		self.fd = fd
		self.start = offset
		self.length = length
		self.block = block
		self.binwalk = binwalk
		self.plugins = plugins
		self.total_read = 0
		self.fd_open = False

		if file_name is None and self.fd is None:
			raise Exception("Entropy.__init__ requires at least the file_name or fd options")

		if self.fd is None:
			self.fd = open(file_name, 'rb')
			self.fd_open = True

		if not self.length:
			self.length = None

		if not self.start:
			self.start = 0

		if not self.block:
			self.block = self.DEFAULT_BLOCK_SIZE
			
		# Some file descriptors aren't seekable (stdin, for example)
		try:
			self.fd.seek(self.start)
		except:
			self.fd.read(self.start)

		if self.binwalk:
			# Set the total_scanned and scan_length values for plugins and status display messages
			self.binwalk.total_scanned = 0
			if self.length:
				self.binwalk.scan_length = self.length
			else:
				self.binwalk.scan_length = common.file_size(self.fd.name) - self.start

	def __enter__(self):
		return self

	def __del__(self):
		self.cleanup()

	def __exit__(self, t, v, traceback):
		self.cleanup()

	def cleanup(self):
		'''
		Clean up any open file objects.
		Called internally by __del__ and __exit__.

		Returns None.
		'''
		try:
			if self.fd_open:
				self.fd.close()
		except:
			pass

	def _read_block(self):
		offset = self.total_read

		if self.length is not None and (self.total_read + self.block) > self.length:
			read_size = self.length - self.total_read
		else:
			read_size = self.block

		data = self.fd.read(read_size)
		dlen = len(data)

		if self.binwalk:
			self.binwalk.total_scanned = self.total_read

		self.total_read += dlen

		return (dlen, data, offset+self.start)

	def gzip(self, offset, data, truncate=True):
		'''
		Performs an entropy analysis based on zlib compression ratio.
		This is the default analysis used as it is faster than the shannon entropy analysis
		and produces basically the same data.
		'''
		# Entropy is a simple ratio of: <zlib compressed size> / <original size>
		e = float(float(len(zlib.compress(data, 9))) / float(len(data)))

		if truncate and e > 1.0:
			e = 1.0

		return e

	def shannon(self, offset, data):
		'''
		Performs a Shannon entropy analysis on a given block of data.
		'''
		entropy = 0
		dlen = len(data)

		if not data:
			return 0

		for x in range(256):
			p_x = float(data.count(chr(x))) / dlen
			if p_x > 0:
				entropy += - p_x*math.log(p_x, 2)

		return (entropy / 8)

	def _do_analysis(self, algorithm):
		'''
		Performs an entropy analysis using the provided algorithm.

		@algorithm - A function/method to call which returns an entropy value.

		Returns a tuple of ([x-coordinates], [y-coordinates], average_entropy), where:

			o x-coordinates = A list of offsets analyzed inside the data.
			o y-coordinates = A corresponding list of entropy for each offset.
		'''
		offsets = []
		entropy = []
		average = 0
		total = 0
		self.total_read = 0
		plug_ret = plugins.PLUGIN_CONTINUE
		plug_pre_ret = plugins.PLUGIN_CONTINUE

		if self.plugins:
			plug_pre_ret = self.plugins._pre_scan_callbacks(self.fd)

		while not ((plug_pre_ret | plug_ret) & plugins.PLUGIN_TERMINATE):
			(dlen, data, offset) = self._read_block()
			if not dlen or not data:
				break

			e = algorithm(offset, data)

			results = {'description' : '%f' % e, 'offset' : offset}

			if self.plugins:
				plug_ret = self.plugins._scan_callbacks(results)
				offset = results['offset']
				e = float(results['description'])

			if not ((plug_pre_ret | plug_ret) & (plugins.PLUGIN_TERMINATE | plugins.PLUGIN_NO_DISPLAY)):
				if self.binwalk:
					self.binwalk.display.results(offset, [results])

				entropy.append(e)
				offsets.append(offset)
				total += e

		try:
			# This results in a divide by zero if one/all plugins returns PLUGIN_TERMINATE or PLUGIN_NO_DISPLAY,
			# or if the file being scanned is a zero-size file.
			average = float(float(total) / float(len(offsets)))
		except:
			pass

		if self.plugins:
			self.plugins._post_scan_callbacks(self.fd)
	
		return (offsets, entropy, average)

	def analyze(self, algorithm=None):
		'''
		Performs an entropy analysis of the data using the specified algorithm.

		@algorithm - A method inside of the Entropy class to invoke for entropy analysis.
			     Default method: self.gzip.
			     Other available methods: self.shannon.
			     May also be a string: 'shannon'.

		Returns the return value of algorithm.
		'''
		algo = self.gzip

		if algorithm:
			if callable(algorithm):
				algo = algorithm

			try:
				if algorithm.lower() == 'shannon':
					algo = self.shannon
			except:
				pass

		data = self._do_analysis(algo)

		return data
	
	def plot(self, x, y, average=0, file_results=[], show_legend=True, save=False):
		'''
		Plots entropy data.

		@x            - List of graph x-coordinates (i.e., data offsets).
		@y            - List of graph y-coordinates (i.e., entropy for each offset).
		@average      - The average entropy.
		@file_results - A list of tuples containing additional analysis data, as returned by Binwalk.single_scan.
		@show_legend  - Set to False to not generate a color-coded legend and plotted x coordinates for the graph.
		@save         - If set to True, graph will be saved to disk rather than displayed.

		Returns None.
		'''
		import matplotlib.pyplot as plt
		import numpy as np

		i = 0
		trigger = 0
		new_ticks = []
		colors = ['darkgreen', 'blueviolet', 'saddlebrown', 'deeppink', 'goldenrod', 'olive', 'black']
		color_mappings = {}

		plt.clf()

		if not file_results and show_legend and average:
			file_results = []

			# Typically the average entropy is used as the trigger level for rising/falling entropy edges.
			# If the average entropy is too low, false rising and falling edges will be marked; if this is
			# the case, and if there is at least one data point greater than ENTROPY_MAX, use ENTROPY_TRIGGER
			# as the trigger level to avoid false edges.
			if average < self.ENTROPY_TRIGGER:
				for point in y:
					if point > self.ENTROPY_MAX:
						trigger = self.ENTROPY_TRIGGER
						break

			if not trigger:
				trigger = average

			for j in range(0, len(x)):
				if j > 0:
					if y[j] >= trigger and y[j-1] < trigger:
						file_results.append((x[j], [{'description' : 'Entropy rising edge'}]))
					elif y[j] <= trigger and y[j-1] > trigger:
						file_results.append((x[j], [{'description' : 'Entropy falling edge'}]))

		if file_results:
			for (offset, results) in file_results:
				label = None
				description = results[0]['description'].split(',')[0]

				if not color_mappings.has_key(description):
					if show_legend:
						label = description

					color_mappings[description] = colors[i]
					i += 1
					if i >= len(colors):
						i = 0
			
				plt.axvline(x=offset, label=label, color=color_mappings[description], linewidth=1.5)
				new_ticks.append(offset)

			if show_legend:
				plt.legend()

				if new_ticks:
					new_ticks.sort()
					plt.xticks(np.array(new_ticks), new_ticks)

		plt.plot(x, y, linewidth=1.5)

		if average:
			plt.plot(x, [average] * len(x), linestyle='--', color='r')

		plt.xlabel('Offset')
		plt.ylabel('Entropy')
		plt.title(self.fd.name)
		plt.ylim(0, 1.5)
		if save:
			plt.savefig(common.unique_file_name(os.path.join(os.path.dirname(self.fd.name), '_' + os.path.basename(self.fd.name)), self.FILE_FORMAT))
		else:
			plt.show()

class Entropy(object):
	'''
	Class for analyzing and plotting data entropy for multiple files.

	A simple example of performing a binwalk scan and overlaying the binwalk scan results on the
	resulting entropy analysis graph:

		import sys
		import binwalk

		bwalk = binwalk.Binwalk()
		scan_results = bwalk.scan(sys.argv[1])

                with binwalk.entropy.Entropy(scan_results, bwalk) as e:
                        e.analyze()

		bwalk.cleanup()
	'''

	DESCRIPTION = "ENTROPY"
	ENTROPY_SCAN = 'entropy'

	def __init__(self, files, binwalk=None, offset=0, length=0, block=0, plot=True, legend=True, save=False, algorithm=None, load_plugins=True, whitelist=[], blacklist=[]):
		'''
		Class constructor.

		@files        - A dictionary containing file names and results data, as returned by Binwalk.scan.
		@binwalk      - An instance of the Binwalk class.
		@offset       - The offset into the data to begin analysis.
		@length       - The number of bytes to analyze.
		@block        - The size of the data blocks to analyze.
		@plot         - Set to False to disable plotting.
		@legend       - Set to False to exclude the legend and custom offset markers from the plot.
		@save         - Set to True to save plots to disk instead of displaying them.
		@algorithm    - Set to 'shannon' to use shannon entropy algorithm.
		@load_plugins - Set to False to disable plugin callbacks.
		@whitelist    - A list of whitelisted plugins.
		@blacklist    - A list of blacklisted plugins.

		Returns None.
		'''
		self.files = files
		self.binwalk = binwalk
		self.offset = offset
		self.length = length
		self.block = block
		self.plot = plot
		self.legend = legend
		self.save = save
		self.algorithm = algorithm
		self.plugins = None
		self.load_plugins = load_plugins
		self.whitelist = whitelist
		self.blacklist = blacklist

		if len(self.files) > 1:
			self.save = True

		if self.binwalk:
			self.binwalk.scan_type = self.binwalk.ENTROPY

	def __enter__(self):
		return self

	def __exit__(self, t, v, traceback):
		return None

	def __del__(self):
		return None

	def set_entropy_algorithm(self, algorithm):
		'''
		Specify a function/method to call for determining data entropy.

		@algorithm - The function/method to call. This will be  passed two arguments:
			     the file offset of the data block, and a data block (type 'str').
		             It must return a single floating point entropy value from 0.0 and 1.0, inclusive.

		Returns None.
		'''
		self.algorithm = algorithm

	def analyze(self):
		'''
		Perform an entropy analysis on the target files.

		Returns a dictionary of:
			
			{
				'file_name' : ([list, of, offsets], [list, of, entropy], average_entropy)
			}
		'''
		results = {}

		if self.binwalk and self.load_plugins:
			self.plugins = plugins.Plugins(self.binwalk, whitelist=self.whitelist, blacklist=self.blacklist)

		for (file_name, overlay) in self.files.iteritems():

			if self.plugins:
				self.plugins._load_plugins()

			if self.binwalk:
				self.binwalk.display.header(file_name=file_name, description=self.DESCRIPTION)

			with FileEntropy(file_name=file_name, binwalk=self.binwalk, offset=self.offset, length=self.length, block=self.block, plugins=self.plugins) as e:
				(x, y, average) = e.analyze(self.algorithm)
				
				if self.plot or self.save:
					e.plot(x, y, average, overlay, self.legend, self.save)
				
				results[file_name] = (x, y, average)

			if self.binwalk:
				self.binwalk.display.footer()

		if self.plugins:
			del self.plugins
			self.plugins = None

		return results

