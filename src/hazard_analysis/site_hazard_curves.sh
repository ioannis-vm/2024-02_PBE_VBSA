#!/usr/bin/bash

# Generate site-specific hazard curves by calling HazardCurveCalc.java

longitude=$(cat data/study_vars/longitude)
latitude=$(cat data/study_vars/latitude)
vs30=$(cat data/study_vars/vs30)

cd src

# make a directory to store the output
# (if it does not exist)
site_hazard_path="../results/site_hazard/"
mkdir -p $site_hazard_path

# download the .jar file (if it is not there)
jar_file_path="../external_tools/opensha-all.jar"
if [ -f "$jar_file_path" ]; then
    echo "opensha-all.jar already exists."
else
    echo "opensha-all.jar does not exist. Downloading."
    wget -P ../external_tools/ "http://opensha.usc.edu/apps/opensha/nightlies/latest/opensha-all.jar"
fi

# compile java code if it has not been compiled already
javafile_path="hazard_analysis/HazardCurveCalc.class"
if [ -f "$javafile_path" ]; then
    echo "HazardCurveCalc.class already compiled"
else
    echo "Compiling HazardCurveCalc.class."
    javac -classpath $jar_file_path:. HazardCurveCalc.java
fi

# obtain hazard curve data
periods=(0.010 0.020 0.030 0.050 0.075 0.10 0.15 0.20 0.25 0.30 0.40 0.50 0.75 1.0 1.5 2.0 3.0 4.0 5.0 7.5 10.0)
curve_names=(0p01 0p02 0p03 0p05 0p075 0p1 0p15 0p2 0p25 0p3 0p4 0p5 0p75 1p0 1p5 2p0 3p0 4p0 5p0 7p5 10p0)

# run calculations in batches
for idx in {0..10}
do
    for idxadd in {0..1}
    do
	subidx=$(echo $idx*2 + $idxadd | bc)
	java -classpath $jar_file_path:hazard_analysis HazardCurveCalc ${periods[$subidx]} $latitude $longitude $vs30 "$site_hazard_path""${curve_names[$subidx]}.txt" &
    done
    wait
done
