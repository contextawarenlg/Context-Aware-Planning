for ses in "all" "juans" "bens"
do
    echo "===================================================================================================================="
    echo "Season: $ses"

    echo "Creating Player & Team Popularity Data..."
    python3 utils/calc_popularity_player.py -season $ses
    python3 utils/calc_popularity_team.py -season $ses

    echo "Extracting Concepts from Gold Summaries..."
    python3 utils/extract_from_gold.py -season $ses

    echo "Creating Imp Player Classifier Data..."
    python3 utils/imp_players_clf_data.py -season $ses

    echo "Creating Imp Player Classifier..."
    python3 player_clf/main.py -season $ses

    echo "Creating Case-Base for ${ses} season..."
    for players in 'imp'
    do
        for ftrs in 'set'
        do 
            python3 create_cb.py -side both -season $ses -players $players -ftrs $ftrs
            python3 create_cb.py -side both -season $ses -players $players -ftrs $ftrs -pop 
            python3 create_cb.py -side both -season $ses -players $players -ftrs $ftrs -tpop 
            python3 create_cb.py -side both -season $ses -players $players -ftrs $ftrs -week
            python3 create_cb.py -side both -season $ses -players $players -ftrs $ftrs -stands
            python3 create_cb.py -side prob -season $ses -players $players -ftrs $ftrs -stands -week -pop -tpop
        done
    done

    echo "Learning feature weights for ${ses} season..."
    python3 utils/feature_weighting.py
    python3 utils/feature_weighting.py -pop
    python3 utils/feature_weighting.py -tpop
    python3 utils/feature_weighting.py -week
    python3 utils/feature_weighting.py -stands
    python3 utils/feature_weighting.py -stands -week -pop -tpop

    echo "Generating from Case-Base for ${ses} season..."
    for players in 'imp'
    do
        for ftrs in 'set'
        do
            for reuse in 'median' 
            do 
                for sim in 'euclidean'
                do
                    echo "Generating Concepts on Test Set with reuse: $reuse, sim: $sim, players: $players and ftrs: $ftrs"
                    python3 gen_concepts.py -season $ses -players $players -ftrs $ftrs -reuse $reuse -sim $sim
                    python3 gen_concepts.py -season $ses -players $players -ftrs $ftrs -reuse $reuse -sim $sim -pop
                    python3 gen_concepts.py -season $ses -players $players -ftrs $ftrs -reuse $reuse -sim $sim -tpop
                    python3 gen_concepts.py -season $ses -players $players -ftrs $ftrs -reuse $reuse -sim $sim -week
                    python3 gen_concepts.py -season $ses -players $players -ftrs $ftrs -reuse $reuse -sim $sim -stands
                    python3 gen_concepts.py -season $ses -players $players -ftrs $ftrs -reuse $reuse -sim $sim -stands -week -pop -tpop
                done
            done
        done
    done

    echo "Evaluating..."
    python3 utils/non_rg.py -eoc len -season $ses
    python3 utils/non_rg.py -eoc concepts -season $ses
    python3 utils/non_rg.py -eoc entities -season $ses
    python3 utils/prep_eval_res.py -season $ses

    echo "${ses} season done!!!"
    echo "===================================================================================================================="
    echo " "
done

