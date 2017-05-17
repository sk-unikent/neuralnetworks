<?php
require_once(dirname(__FILE__) . '/lib.php');

$validNets = array();
$diriterator = new RecursiveDirectoryIterator("trained");
$iterator = new RecursiveIteratorIterator($diriterator, RecursiveIteratorIterator::SELF_FIRST);
foreach ($iterator as $file) {
    if (strrpos($file, '.nn') == strlen($file) - 3) {
        $validNets[] = $file;
    }
}

$networkstr = 'trained/maths/basic/xor.nn';
if (isset($_GET['net']) && in_array($_GET['net'], $validNets)) {
    $networkstr = $_GET['net'];
}

$network = KeltyNN\NeuralNetwork::loadfile(dirname(__FILE__) . '/' . $networkstr);
?>
<!DOCTYPE html>
<html>
	<head>
		<title><?php echo $network->getTitle(); ?></title>
		<link href='//cdnjs.cloudflare.com/ajax/libs/vis/4.17.0/vis.min.css' rel='stylesheet' type='text/css'>
        <link href='//cdnjs.cloudflare.com/ajax/libs/chosen/1.6.2/chosen.min.css' rel='stylesheet' type='text/css'>
        <link href='//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css' rel='stylesheet' type='text/css'>
        <link href='//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap-theme.min.css' rel='stylesheet' type='text/css'>
        <script src="//cdnjs.cloudflare.com/ajax/libs/jquery/3.1.1/jquery.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/vis/4.17.0/vis.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/chosen/1.6.2/chosen.jquery.min.js"></script>
        <script src="//maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js"></script>
        <script src="//cdnjs.cloudflare.com/ajax/libs/Chart.js/2.4.0/Chart.bundle.min.js"></script>
	</head>
	<body>
		<div class='container-fluid'>
            <div class="row">
                <div class="col-md-8">
                    <h1><?php echo $network->getTitle(); ?></h1>
                    <form method="GET" action="index.php">
                        <select name="net">
                            <?php
                            foreach ($validNets as $validNet) {
                                $selected = ($validNet == $networkstr) ? ' selected="selected"' : '';
                                echo "<option value=\"{$validNet}\"{$selected}>{$validNet}</option>";
                            }
                            ?>
                        </select>
                    </form>
                    <div id="info">
                        <?php
                        echo '<h3>' . $network->getTitle() . '</h3>';
                        echo '<p>' . $network->getDescription() . '</p>';
                        ?>
                    </div>
                </div>
                <div class="col-md-4">
                    <div style="width: 200px; height:200px; float: right; text-align: right;">
                        <canvas id="activationgraph" width="200" height="200"></canvas>
                        <?php
                        echo '<small>' . $network->getType() . ' (v' . $network->getVersion() . ')</small>';
                        ?>
                    </div>
                </div>
            </div>
            <div class="row">
                <div id="network"></div>
            </div>
		</div>

        <script type="text/javascript">
            // create an array with nodes
            var nodes = new vis.DataSet([
                <?php
                for ($i = 0; $i < $network->getInputCount(); $i++) {
                    echo "{id: '0_{$i}', label: 'Input {$i}'},\n";
                }
                foreach ($network->nodeThreshold as $layer => $nodes) {
                    foreach ($nodes as $num => $node) {
                        echo "{id: '{$layer}_{$num}', label: '{$node}'},\n";
                    }
                }
                ?>
            ]);

            // create an array with edges
            var edges = new vis.DataSet([
            <?php
            foreach ($network->edgeWeight as $layer => $nodes) {
                foreach ($nodes as $from => $weights) {
                    $nextlayer = $layer + 1;
                    foreach ($weights as $to => $weight) {
                        // Support FlexNet based nets.
                        if (is_string($to)) {
                            $strto = $to;
                        } else {
                            $strto = "{$nextlayer}_{$to}";
                        }

                        echo "{from: '{$layer}_{$from}', to: '{$strto}', label: '{$weight}'},\n";
                    }
                }
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
                layout: {
                    hierarchical: {
                        direction: "LR",
                        sortMethod: "directed",
                        levelSeparation: 500,
                        nodeSpacing: 150
                    }
                },
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

            // Select box.
            $('select').chosen().change(function() {
              $(this).parent().submit();
            });

            // Activation graph.
            var chartData = [];

            var ctx = $("#activationgraph");
            var myChart = new Chart(ctx, {
                type: 'line',
                data: {
                    datasets: [{
                        label: 'Activation function',
                        fill: false,
                        borderColor: "#FF7265",
                        data: [
                            <?php
                                $out = array();
                                $xpoints = range(-1, 1, 0.1);
                                foreach ($xpoints as $x) {
                                    $y = $network->activation($x);
                                    $out[] = "{x: {$x}, y: {$y}}";
                                }
                                echo implode(',', $out);
                            ?>
                        ]
                    }, {
                        label: 'Derivative function',
                        fill: false,
                        borderColor: "#86FFBC",
                        data: [
                            <?php
                                $out = array();
                                $xpoints = range(-1, 1, 0.1);
                                foreach ($xpoints as $x) {
                                    $y = $network->derivativeActivation($x);
                                    $out[] = "{x: {$x}, y: {$y}}";
                                }
                                echo implode(',', $out);
                            ?>
                        ]
                    }]
                },
                options: {
                    scales: {
                        xAxes: [{
                            type: 'linear',
                            position: 'bottom'
                        }]
                    }
                }
            });
        </script>
	</body>
</html>
