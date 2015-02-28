#!/bin/bash

rm -f output.html
echo "<table>" >> output.html
echo "<tr><td>&nbsp;</td><td>Original</td><td colspan=2>Gaussian</td><td colspan=2>Linear</td></tr>" >> output.html
echo "<tr><td>&nbsp;</td><td>&nbsp;</td><td>No Non Max Suppression</td><td>Non Max Suppression</td><td>No Non Max Suppression</td><td>Non Max Suppression</td></tr>" >> output.html
for i in `ls -r test_input/*.jpg | grep -v faces`; do
	short_i=${i%.jpg}
	short_i=${short_i#*/}
	echo "<tr>"
	echo "<td>${short_i}</td>"
	echo "<td><img src=\"$i\"/></td>"
	for j in `ls -r test_output/*${short_i}_* 2> /dev/null | grep gaussian`; do
		echo "<td><img src=\"$j\"/></td>"
	done
	for j in `ls -r test_output/*${short_i}_* 2> /dev/null | grep linear`; do
		echo "<td><img src=\"$j\"/></td>"
	done
	echo "</tr>"
done >> output.html
echo "</table>" >> output.html