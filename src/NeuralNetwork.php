<?php
/**
 * Base Neural Network.
 */
namespace KeltyNN;

class NeuralNetwork
{
    protected $type;
    protected $version;
    protected $title;
    protected $description;

    /**
     * Get a title.
     */
    public function getTitle()
    {
        return $this->title;
    }

    /**
     * Set a title.
     */
    public function setTitle($title)
    {
        $this->title = $title;
    }

    /**
     * Get a type.
     */
    public function getType()
    {
        return $this->type;
    }

    /**
     * Get a version.
     */
    public function getVersion()
    {
        return $this->version;
    }

    /**
     * Get a description.
     */
    public function getDescription()
    {
        return $this->description;
    }

    /**
     * Set a description.
     */
    public function setDescription($description)
    {
        $this->description = $description;
    }

    /**
     * Saves a neural network to a file.
     *
     * @param string $filename The filename to save the neural network to
     *
     * @return bool 'true' on success, 'false' otherwise
     */
    public function save($filename)
    {
        $export = $this->export();

        return file_put_contents($filename, serialize($export));
    }

    /**
     * Loads a neural network from a file saved by the 'save()' function. Clears
     * the training and control data added so far.
     *
     * @param array $data The data to load the network from
     *
     * @return bool 'true' on success, 'false' otherwise
     */
    public static function load($data)
    {
        $class = '\\KeltyNN\\Networks\\' . $data['type'];

        $obj = new $class($data['nodeCount']);
        if ($obj->restore($data)) {
            return $obj;
        }

        return false;
    }

    /**
     * Loads a neural network from a file saved by the 'save()' function. Clears
     * the training and control data added so far.
     *
     * @param string $filename The filename to load the network from
     *
     * @return bool 'true' on success, 'false' otherwise
     */
    public static function loadfile($filename)
    {
        if (!file_exists($filename)) {
            return false;
        }

        $data = file_get_contents($filename);
        $data = unserialize($data);
        if (!$data) {
            return false;
        }

        return self::load($data);
    }

    /**
     * Loads a neural network from a file saved by the 'save()' function. Clears
     * the training and control data added so far.
     *
     * @param string $filename The filename to load the network from
     *
     * @return bool 'true' on success, 'false' otherwise
     */
    protected function restore($data)
    {
        // make sure all standard preparations performed
        $this->initWeights();

        $this->weightsInitialized = true;

        // if we do not reset the training and control data here, then we end up
        // with a bunch of IDs that do not refer to the actual data we're training
        // the network with.
        $this->controlInputs = array();
        $this->controlOutput = array();

        $this->trainInputs = array();
        $this->trainOutput = array();

        $this->import($data);

        return true;
    }

    /**
     * Combines two neural networks such that the output nodes of network a
     * are fed into the input nodes of network b.
     */
    public static function combine($networka, $networkb)
    {
        // First, sanity checks.
        if ($networka->getOutputCount() !== $networkb->getInputCount()) {
            throw new \InvalidArgumentException("Number of output nodes on network A must equal number of input nodes on network b!");
        }

        if ($networka->type !== $networkb->type) {
            throw new \InvalidArgumentException("Type of network A must equal type of network b!");
        }

        if ($networka->version !== $networkb->version) {
            throw new \InvalidArgumentException("Version of network A must equal version of network b!");
        }

        // Good! Create a new net.
        $input = $networka->getInputCount();
        $output = $networkb->getOutputCount();

        // Work out hidden layers.
        $hiddena = $networka->getHiddenCounts();
        $hiddenb = $networkb->getHiddenCounts();
        $hidden = array_merge($hiddena, array($networkb->getInputCount()), $hiddenb);

        // Work out total new number of neurons.
        $total = array_merge(array($input), $hidden, array($output));

        // Create new neural network.
        $class = '\\KeltyNN\\Networks\\' . $networka->type;
        $networkc = new $class($total);

        // Now, restore nodes into network C.
        $layernum = 1;
        foreach (array_merge($networka->nodeThreshold, $networkb->nodeThreshold) as $layer) {
            $networkc->nodeThreshold[$layernum] = $layer;
            $layernum++;
        }

        // Now restore interconnects into network C.
        $layernum = 0;
        foreach (array_merge($networka->edgeWeight, $networkb->edgeWeight) as $layer) {
            $networkc->edgeWeight[$layernum] = $layer;
            $layernum++;
        }

        return $networkc;
    }

    /**
     * Adds two neural networks such that the two networks share input nodes
     * but provide their own output nodes.
     */
    public static function add($networka, $networkb)
    {
        // First, sanity checks.
        if ($networka->getInputCount() !== $networkb->getInputCount()) {
            throw new \InvalidArgumentException("Number of input nodes on network A must equal number of input nodes on network b!");
        }

        if ($networka->type !== $networkb->type) {
            throw new \InvalidArgumentException("Type of network A must equal type of network b!");
        }

        if ($networka->version !== $networkb->version) {
            throw new \InvalidArgumentException("Version of network A must equal version of network b!");
        }

        // Good! Create a new net.
        $input = $networka->getInputCount();
        $output = $networka->getOutputCount() + $networkb->getOutputCount();
        $hiddena = $networka->getHiddenCounts();
        $hiddenb = $networkb->getHiddenCounts();

        $hidden = array();
        for ($i = 0; $i < max(count($hiddena), count($hiddenb)); $i++) {
            $hidden[] = $hiddena[$i] + $hiddenb[$i];
        }

        // Work out total new number of neurons.
        $total = array_merge(array($input), $hidden, array($output));

        // Create new neural network.
        $class = '\\KeltyNN\\Networks\\' . $networka->type;
        $networkc = new $class($total);

        // Now, restore nodes into network C.
        $networkc->nodeThreshold = array();
        foreach ($networka->nodeThreshold as $layernum => $layer) {
            foreach ($layer as $node) {
                $networkc->nodeThreshold[$layernum][] = $node;
            }
        }
        foreach ($networkb->nodeThreshold as $layernum => $layer) {
            foreach ($layer as $node) {
                $networkc->nodeThreshold[$layernum][] = $node;
            }
        }

        // TODO - also, fill in any blanks we might need at the end of a net, or support multi-layer edges.
        // If one net is longer than the other.

        // Now restore interconnects into network C.
        $networkc->edgeWeight = array();
        foreach ($networka->edgeWeight as $layernum => $layer) {
            $networkc->edgeWeight[$layernum] = array();
            foreach ($layer as $edge) {
                $networkc->edgeWeight[$layernum][] = $edge;
            }
        }
        foreach ($networkb->edgeWeight as $layernum => $layer) {
            foreach ($layer as $nodenum => $edge) {
                foreach ($edge as $weightnum => $weight) {
                    if ($layernum == 0) {
                        $networkc->edgeWeight[$layernum][$nodenum][] = $weight;
                        continue;
                    }

                    // Where do we start?
                    $nodenumpad = count($networka->edgeWeight[$layernum]);
                    $weightnum += count($networka->nodeThreshold[$layernum + 1]);
                    $networkc->edgeWeight[$layernum][$nodenum + $nodenumpad][$weightnum] = $weight;
                }
            }
        }

        return $networkc;
    }
}
