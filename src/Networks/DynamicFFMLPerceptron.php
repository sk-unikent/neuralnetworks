<?php
/**
 * Multi-layer Neural Network in PHP.
 *
 * Rather than static test data, we have a dynamic feedback mechanism for the test data.
 *
 * @author S.Kelty <S.Kelty@kent.ac.uk>
 * @version 2.0
 * @license http://opensource.org/licenses/bsd-license.php BSD License
 */
namespace KeltyNN\Networks;

class DynamicFFMLPerceptron extends FFMLPerceptron
{
    /**
     * Creates a neural network.
     *
     * Example:
     * <code>
     * // create a network with 4 input nodes, 10 hidden nodes, and 4 output nodes
     * $n = new NeuralNetwork(4, 10, 4);
     *
     * // create a network with 4 input nodes, 1 hidden layer with 10 nodes,
     * // another hidden layer with 10 nodes, and 4 output nodes
     * $n = new NeuralNetwork(4, 10, 10, 4);
     *
     * // alternative syntax
     * $n = new NeuralNetwork(array(4, 10, 10, 4));
     * </code>
     *
     * @param array $nodeCount The number of nodes in the consecutive layers.
     */
    public function __construct($nodeCount)
    {
        if (!is_array($nodeCount)) {
            $nodeCount = func_get_args();
        }
        parent::__construct($nodeCount);

        $this->type = 'FFMLPerceptron'; // Because the result is compatible.
        $this->version = '3.0';
    }

    /**
     * Start the training process.
     *
     * @param int   $maxEpochs The maximum number of epochs
     * @param float $maxError  The maximum squared error in the training data
     *
     * @return bool 'true' if the training was successful, 'false' otherwise
     */
    public function train($callback, $maxEpochs = 500, $maxError = 0.01)
    {
        if (!$this->weightsInitialized) {
            $this->initWeights();
        }

        if ($this->isVerbose()) {
            echo '<table>';
            echo '<tr><th>#</th><th>error(trainingdata)</th><th>error(controldata)</th><th>slope(error(controldata))</th></tr>';
        }

        $epoch = 0;
        $errorControlSet = array();
        $avgErrorControlSet = array();
        $sample_count = 10;
        do {
            for ($i = 0; $i < count($this->trainInputs); $i++) {
                // select a training pattern at random
                $index = mt_rand(0, count($this->trainInputs) - 1);

                // determine the input, and the desired output
                $input = $this->trainInputs[$index];

                // calculate the actual output
                $output = $this->calculate($input);

                // calculate desired output
                $desired_output = $callback($input, $output);

                // change network weights
                $this->backpropagate($output, $desired_output);
            }

            // buy some time
            set_time_limit(300);

            //display the overall network error after each epoch
            $squaredError = $this->squaredErrorEpoch();
            if ($epoch % 2 == 0) {
                $squaredErrorControlSet = $this->squaredErrorControlSet();
                $errorControlSet[] = $squaredErrorControlSet;

                if (count($errorControlSet) > $sample_count) {
                    $avgErrorControlSet[] = array_sum(array_slice($errorControlSet, -$sample_count)) / $sample_count;
                }

                list($slope, $offset) = $this->fitLine($avgErrorControlSet);
                $controlset_msg = $squaredErrorControlSet;
            } else {
                $controlset_msg = '';
            }

            if ($this->isVerbose()) {
                echo "<tr><td><b>$epoch</b></td><td>$squaredError</td><td>$controlset_msg";
                echo "<script type='text/javascript'>window.scrollBy(0,100);</script>";
                echo "</td><td>$slope</td></tr>";
                echo '</td></tr>';

                flush();
                ob_flush();
            }

            // conditions for a 'successful' stop:
            // 1. the squared error is now lower than the provided maximum error
            $stop_1 = $squaredError <= $maxError || $squaredErrorControlSet <= $maxError;

            // conditions for an 'unsuccessful' stop
            // 1. the maximum number of epochs has been reached
            $stop_2 = $epoch++ > $maxEpochs;

            // 2. the network's performance on the control data is getting worse
            $stop_3 = $slope > 0;
        } while (!$stop_1 && !$stop_2 && !$stop_3);

        $this->setEpoch($epoch);
        $this->setErrorTrainingSet($squaredError);
        $this->setErrorControlSet($squaredErrorControlSet);
        $this->setTrainingSuccessful($stop_1);

        if ($this->isVerbose()) {
            echo '</table>';
        }

        return $stop_1;
    }
}
