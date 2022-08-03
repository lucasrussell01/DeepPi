#!/bin/bash

export directory=$1
export output_prefix=$2
export samples=$3


targets=($(xrdfs gfe02.grid.hep.ph.ic.ac.uk:1097 ls $directory )) 

echo $targets
for i in "${targets[@]}"; do
  names=($(xrdfs gfe02.grid.hep.ph.ic.ac.uk:1097 ls /$i | cut -d"/" -f2-))
  for j in "${names[@]}"; do
    name=$(echo $j | cut -d"/" -f6-| cut -d"_" -f2-)
    type="MC"
    if [[ $name == "Tau"* || "$name" == "MuonEG"* || "$name" == "SingleElectron"* || "$name" == "SingleMuon"* || "$name" == "EGamma"* || "$name" == "DoubleMuon"* ]]; then
      type="Data"
    fi
    if [ "$samples" ]; then 
      if [[ $name != *"$samples"* ]]; then continue; fi
    fi
    echo "Getting filelist for : " $name

    file_name=$output_prefix"_"$type"_106X_"$name".dat"
    date_names=($(xrdfs gfe02.grid.hep.ph.ic.ac.uk:1097 ls $j )) # | cut -d"/" -f2-))
    export most_recent=0
    for k in "${date_names[@]}"; do
      date=$(echo $k | rev | cut -d"/" -f1 | rev | cut -d"_" -f1)$(echo $k | rev | cut -d"/" -f1 | rev | cut -d"_" -f2-)
      if [[ $date > $most_recent ]]; then
        export most_recent_string=$k
        most_recent=$date
      fi
    done
    count=0
    directories=($(xrdfs gfe02.grid.hep.ph.ic.ac.uk:1097 ls $most_recent_string))

    for l in "${directories[@]}"; do
      if [[ $count == 0 ]]; then
        xrdfs gfe02.grid.hep.ph.ic.ac.uk:1097 ls $l/ | grep .root | cut -d"/" -f5- > $file_name
      else
        xrdfs gfe02.grid.hep.ph.ic.ac.uk:1097 ls $l/ | grep .root | cut -d"/" -f5- >> $file_name
      fi
      ((count++))
    done
  done
done

# tar -czf $output_prefix"_filelists.tgz" $output_prefix*.dat