#!/usr/bin/bash

# Perform seismic hazard deaggregation using DisaggregationCalc.java
# and get GMM mean and stdev results using GMMCalc.java

longitude=$(cat data/study_vars/longitude)
latitude=$(cat data/study_vars/latitude)
vs30=$(cat data/study_vars/vs30)


site_hazard_path="results/site_hazard/"

# get the codes of the archetypes of this study
arch_codes="smrf_3_ii smrf_3_iv smrf_6_ii smrf_6_iv smrf_9_ii smrf_9_iv scbf_3_ii scbf_3_iv scbf_6_ii scbf_6_iv scbf_9_ii scbf_9_iv brbf_3_ii brbf_3_iv brbf_6_ii brbf_6_iv brbf_9_ii brbf_9_iv"

# download the .jar file (if it is not there)
jar_file_path="external_tools/opensha-all.jar"
if [ -f "$jar_file_path" ]; then
    echo "The file exists."
else
    echo "The file does not exist. Downloading file."
    wget -P external_tools/ "http://opensha.usc.edu/apps/opensha/nightlies/latest/opensha-all.jar"
fi

# compile java code if it has not been compiled already
javafile_path="src/hazard_analysis/DisaggregationCalc.class"
if [ -f "$javafile_path" ]; then
    echo "Already compiled DisaggregationCalc"
else
    echo "Compiling DisaggregationCalc.java"
    javac -classpath $jar_file_path src/hazard_analysis/DisaggregationCalc.java
fi

# javafile_path="src/hazard_analysis/GMMCalc.class"
# if [ -f "$javafile_path" ]; then
#     echo "Already compiled GMMCalc"
# else
#     echo "Compiling GMMCalc.java"
#     javac -classpath $jar_file_path src/hazard_analysis/GMMCalc.java
# fi

for code in $arch_codes
do
    
    # Get the period of that archetype
    period=$(cat data/$code/period_closest)

    # Get the hazard level midpoint Sa's
    mapes=$(awk -F, '{if (NR!=1) {print $6}}' results/site_hazard/Hazard_Curve_Interval_Data.csv)

    i=1
    for mape in $mapes
    do

	# perform seismic hazard deaggregation
	mkdir -p results/$code/site_hazard
	sa=$(python -m src.hazard_analysis.interp_uhs --period $period --mape $mape)
	java -classpath $jar_file_path:src/hazard_analysis DisaggregationCalc $period $latitude $longitude $vs30 $sa results/$code/site_hazard/deaggregation_$i.txt

	# # generate GMM results: not required when using CS_Selection for ground motion selection
	# mbar=$(cat ../results/$code/site_hazard/deaggregation_$i.txt | grep Mbar | awk '{print $3}')
	# dbar=$(cat ../results/$code/site_hazard/deaggregation_$i.txt | grep Dbar | awk '{print $3}')
	# java -classpath $jar_file_path:src/hazard_analysis GMMCalc $mbar $dbar $vs30 ../results/$code/site_hazard/gmm_$i.txt

	i=$(($i+1))	

    done
        
done
