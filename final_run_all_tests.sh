###
### ENGLISH
###

## GREEDY
#bash alternate_evaluate.sh english 9 &

## BEAM SEARCH (param = number of beams)
#bash alternate_evaluate.sh english 9 "-b 2" b-2 &
#bash alternate_evaluate.sh english 9 "-b 4" b-4 &
#bash alternate_evaluate.sh english 9 "-b 8" b-8 &
#bash alternate_evaluate.sh english 9 "-b 12" b-12 &
#bash alternate_evaluate.sh english 9 "-b 16" b-16 &
#bash alternate_evaluate.sh english 9 "-b 24" b-24 &
#bash alternate_evaluate.sh english 9 "-b 32" b-32 &

## DYNAMIC BEAM SEARCH (param = % of top-beam-score to cutoff at)
#bash alternate_evaluate.sh english 9 "-D -b 50" db-50 &
#bash alternate_evaluate.sh english 9 "-D -b 60" db-60 &
#bash alternate_evaluate.sh english 9 "-D -b 70" db-70 &
#bash alternate_evaluate.sh english 9 "-D -b 80" db-80 &
#bash alternate_evaluate.sh english 9 "-D -b 90" db-90 &
#bash alternate_evaluate.sh english 9 "-D -b 99" db-99 &

## HEURISTIC BACKTRACKING (param = # backtracks)
#bash alternate_evaluate.sh english 9 "-B 2" hb-2 &
#bash alternate_evaluate.sh english 9 "-B 4" hb-4 &
#bash alternate_evaluate.sh english 9 "-B 8" hb-8 &
#bash alternate_evaluate.sh english 9 "-B 12" hb-12 &
#bash alternate_evaluate.sh english 9 "-B 16" hb-16 &
#bash alternate_evaluate.sh english 9 "-B 24" hb-24 &
#bash alternate_evaluate.sh english 9 "-B 32" hb-32 &
#bash alternate_evaluate.sh english 9 "-B 48" hb-48 &
#bash alternate_evaluate.sh english 9 "-B 64" hb-64 &

## SELECTIONAL BRANCHING (param = neg log prob needed to spawn new branch)
#bash alternate_evaluate.sh english 9 "-b 32 -M .105" sb-90 &
#bash alternate_evaluate.sh english 9 "-b 32 -M .223" sb-80 &
#bash alternate_evaluate.sh english 9 "-b 32 -M .356" sb-70 &
#bash alternate_evaluate.sh english 9 "-b 32 -M .510" sb-60 &
#bash alternate_evaluate.sh english 9 "-b 32 -M .693" sb-50 &

## HEURISTIC BACKTRACKING w/ CUTOFF (param = max # backtracks)
#bash alternate_evaluate.sh english 9 "-B 2 --hb_cutoff" hbc-2 &
#bash alternate_evaluate.sh english 9 "-B 4 --hb_cutoff" hbc-4 &
#bash alternate_evaluate.sh english 9 "-B 8 --hb_cutoff" hbc-8 &
#bash alternate_evaluate.sh english 9 "-B 12 --hb_cutoff" hbc-12 &
#bash alternate_evaluate.sh english 9 "-B 16 --hb_cutoff" hbc-16 &
#bash alternate_evaluate.sh english 9 "-B 24 --hb_cutoff" hbc-24 &
#bash alternate_evaluate.sh english 9 "-B 32 --hb_cutoff" hbc-32 &
#bash alternate_evaluate.sh english 9 "-B 48 --hb_cutoff" hbc-48 &
#bash alternate_evaluate.sh english 9 "-B 64 --hb_cutoff" hbc-64 &

###
### CHINESE
###

#bash alternate_evaluate.sh chinese g &

#bash alternate_evaluate.sh chinese g "-b 2" b-2 &
#bash alternate_evaluate.sh chinese g "-b 4" b-4 &
#bash alternate_evaluate.sh chinese g "-b 8" b-8 &
#bash alternate_evaluate.sh chinese g "-b 12" b-12 &
#bash alternate_evaluate.sh chinese g "-b 16" b-16 &
#bash alternate_evaluate.sh chinese g "-b 24" b-24 &
#bash alternate_evaluate.sh chinese g "-b 32" b-32 &

#bash alternate_evaluate.sh chinese g "-D -b 50" db-50 &
#bash alternate_evaluate.sh chinese g "-D -b 60" db-60 &
#bash alternate_evaluate.sh chinese g "-D -b 70" db-70 &
#bash alternate_evaluate.sh chinese g "-D -b 80" db-80 &
#bash alternate_evaluate.sh chinese g "-D -b 90" db-90 &
#bash alternate_evaluate.sh chinese g "-D -b 99" db-99 &

#bash alternate_evaluate.sh chinese g "-B 2" hb-2 &
#bash alternate_evaluate.sh chinese g "-B 4" hb-4 &
#bash alternate_evaluate.sh chinese g "-B 8" hb-8 &
#bash alternate_evaluate.sh chinese g "-B 12" hb-12 &
#bash alternate_evaluate.sh chinese g "-B 16" hb-16 &
#bash alternate_evaluate.sh chinese g "-B 24" hb-24 &
#bash alternate_evaluate.sh chinese g "-B 32" hb-32 &
#bash alternate_evaluate.sh chinese g "-B 48" hb-48 &
#bash alternate_evaluate.sh chinese g "-B 64" hb-64 &

#bash alternate_evaluate.sh chinese g "-b 32 -M .105" sb-90 &
#bash alternate_evaluate.sh chinese g "-b 32 -M .223" sb-80 &
#bash alternate_evaluate.sh chinese g "-b 32 -M .356" sb-70 &
#bash alternate_evaluate.sh chinese g "-b 32 -M .510" sb-60 &
#bash alternate_evaluate.sh chinese g "-b 32 -M .693" sb-50 &

#bash alternate_evaluate.sh chinese 9 "-B 2 --hb_cutoff" hbc-2 &
#bash alternate_evaluate.sh chinese 9 "-B 4 --hb_cutoff" hbc-4 &
#bash alternate_evaluate.sh chinese 9 "-B 8 --hb_cutoff" hbc-8 &
#bash alternate_evaluate.sh chinese 9 "-B 12 --hb_cutoff" hbc-12 &
#bash alternate_evaluate.sh chinese 9 "-B 16 --hb_cutoff" hbc-16 &
#bash alternate_evaluate.sh chinese 9 "-B 24 --hb_cutoff" hbc-24 &
#bash alternate_evaluate.sh chinese 9 "-B 32 --hb_cutoff" hbc-32 &
#bash alternate_evaluate.sh chinese 9 "-B 48 --hb_cutoff" hbc-48 &
#bash alternate_evaluate.sh chinese 9 "-B 64 --hb_cutoff" hbc-64 &


