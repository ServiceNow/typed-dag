for mechanism in linear anm nn  #; mechanisms used in the SCM
do
for n_nodes_test in 50 # 10 number of nodes in the DAG
do
for n_test in 10000 # sample size
do
for n_types in 2 3 4 5 # number of types
do
for prob_inter in 0.3 0.7 # determine the prob of linking two nodes of different types
do
for prob_intra in 0 # determine the prob of linking two nodes of the same type
do
	(python3.7 main.py --mechanism ${mechanism} \
		--suffix ${mechanism}_type${n_types}_pinter${prob_inter}_pintra${prob_intra} \
		--prob-inter ${prob_inter} \
		--prob-intra ${prob_intra} \
		--noise-coeff 0.1 \
		--n-datasets 10 \
		--n-types ${n_types} \
		--n-dags-train 1 \
		--n-dags-test 1 \
		--n-train 1000 \
		--n-test ${n_test} \
		--n-nodes-test 10 \
		--n-interventions-test ${n_nodes_test} \
		--intervention-train \
		--intervention-test \
		--n-nodes-test ${n_nodes_test}) &
done
done
done
done
done
done
wait
