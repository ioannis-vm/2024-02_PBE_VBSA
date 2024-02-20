mkdir -p results/design_logs
rm -r results/design_logs/*
for code in smrf scbf brbf
do
    for stor in 3 6 9
    do
		for rc in ii iv
		do
			python -m src.structural_analysis.design.design_"$code"_"$stor"_"$rc" >> "results/design_logs/""$code"_"$stor"_"$rc".txt
		done
    done
done
