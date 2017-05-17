<?php
/**
 * Training script.
 */
require_once(dirname(__FILE__) . '/../lib.php');

for ($hidden = 3; $hidden < 20; $hidden++) {
foreach (array(
    'KeltyNN\\Networks\\FFMLPerceptron',
    'KeltyNN\\Networks\\FFMLRELUPerceptron',
    'KeltyNN\\Networks\\FFMLLeakyRELUPerceptron',
    'KeltyNN\\Networks\\FFMLLinearPerceptron',
    'KeltyNN\\Networks\\FFMLHyperbolicPerceptron',
    'KeltyNN\\Networks\\FFMLSigPerceptron',
    'KeltyNN\\Networks\\FFMLBentIdentPerceptron'
) as $classname) {
    $time = microtime(True);
    // Create a Perceptron network.
    $n = new $classname(2, $hidden, 1);
    $n->setTitle('Equals function');
    $n->setDescription('Given two positive inputs, returns 1 if they are equal.');
    $n->setVerbose(false);

    // Add test-data to the network.
    for ($i = 0; $i < 10; $i++) {
        $n->addTestData(array(((float)"0.$i"), ((float)"0.$i")), array(1));
        $n->addTestData(array(((float)"0.$i"), ((float)"0.$i") + 0.1), array(0));
        $n->addTestData(array(((float)"0.$i"), ((float)"0.$i") - 0.1), array(0));
        //$n->addTestData(array($i, $i + rand(1, 3000)), array(0));
        //if ($i >= 2) {
        //    $n->addTestData(array($i, $i / 2), array(0));
        //}
    }
    //$n->addTestData(array(1, 0), array(1));
    //$n->addTestData(array(1, 1), array(0));
    //$n->addTestData(array(0, 1), array(1));
    //$n->addTestData(array(0, 0), array(0));

    // we try training the network for at most $max times
    $max = 1;
    $j = 0;

    // Train the network.
    while (!($success = $n->train(2000, 0.1)) && ++$j < $max) {
    }

    $epochs = $n->getEpoch();
    $time = microtime(True) - $time;
    if ($success) {
        echo "{$classname} - Success in {$epochs} training rounds over {$j} attempts in {$time}s.\n";

        $n->save(dirname(__FILE__) . '/../trained/maths/advanced/eq_pos.nn');
        exit(0);
    } else {
        echo "{$classname} - failed after {$epochs} training rounds over {$time}s.\n";
    }
}
}
