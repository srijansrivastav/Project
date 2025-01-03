#!/usr/bin/env python

import sys
import re

for line in sys.stdin:
	cols = line.split('\t')
	if re.match('^\\d{1,9}$', cols[4]):
		print '%s\t%s,%s' % (cols[2], cols[1], cols[4])
