<?php
require_once(dirname(__FILE__) . '/lib.php');

$nodes = 14;
$edges = array(
    array('from' => 1, 'to' => 2),
    array('from' => 1, 'to' => 3),
    array('from' => 2, 'to' => 4),
    array('from' => 4, 'to' => 5),
    array('from' => 6, 'to' => 5),
    array('from' => 5, 'to' => 6),
    array('from' => 3, 'to' => 6),
    array('from' => 6, 'to' => 7),
    array('from' => 5, 'to' => 8),
    array('from' => 10, 'to' => 9),
    array('from' => 7, 'to' => 10),
    array('from' => 9, 'to' => 11),
    array('from' => 8, 'to' => 11),
    array('from' => 6, 'to' => 11),
    array('from' => 11, 'to' => 12),
    array('from' => 12, 'to' => 13)
);

$network = new KeltyNN\Networks\DynamicFFMLPerceptron(1, 4, 8, 4, 1);
$network->setTitle('Pathfinder NN');
$network->setDescription('Given a current node, outputs the best node to jump to next.');
$network->setVerbose(false);

// Add test-data to the network.
for ($i = 1; $i < $nodes; $i++) {
    $n->addTestData(array($i), array());
}

$callbackfunc = function($input, $result) {
    // TODO - Weight the choice based on the distance from the end.
    $result = round($result);
    foreach ($edges as $edge) {
        if ($edge['from'] == $result) {
            return $edge['to'];
        }

        // TODO - last best result?
    }
};

while (!($success = $n->train($callbackfunc, 1000, 0.001)) && ++$j < $max) {
}

$network->save(dirname(__FILE__) . '/trained/pathfinding/simple.nn');
?>
<html>
	<head>
		<title><?php //echo $network->getTitle(); ?></title>
		<link href='//cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis.min.css' rel='stylesheet' type='text/css'>
	</head>
	<body>
		<div class='container'>
            <h1><?php //echo $network->getTitle(); ?></h1>
            <div id="network"></div>
		</div>

        <script src="//cdnjs.cloudflare.com/ajax/libs/vis/4.16.1/vis.min.js"></script>

        <script type="text/javascript">
          // create an array with nodes
          var nodes = new vis.DataSet([
              <?php
              for ($i = 1; $i < $nodes; $i++) {
                  echo "{id: '{$i}', label: 'Input {$i}'},\n";
              }
              ?>
          ]);

          // create an array with edges
          var edges = new vis.DataSet([
              <?php
              foreach ($edges as $arr) {
                  $from = $arr['from'];
                  $to = $arr['to'];
                  echo "{from: '{$from}', to: '{$to}'},\n";
              }
              ?>
          ]);

          // create a network
          var container = document.getElementById('network');
          var data = {
            nodes: nodes,
            edges: edges
          };
            var options = {
                physics: {
                    enabled: false
                },
                nodes: {
                    shape: 'circle'
                },
                edges: {
                    smooth: false,
                    arrows: {
                        to : true
                    }
                },
                configure: true
            };
          var network = new vis.Network(container, data, options);
        </script>
	</body>
</html>
