<?php

namespace KeltyNN\Trainers;

class Genetic implements Trainer
{
    const MAX_GENERATIONS = 10;
    const MAX_SPECIES = 10;
    const MAX_PER_SPECIES = 100;
    const BEST_PERC = 0.05;

    protected $basenet;
    protected $bestnet;
    protected $bestscore;
    protected $species;
    protected $genes;
    protected $generation;

    public function __construct($network) {
        $this->basenet = $network;
        $this->bestnet = $network;
        $this->species = array();
        $this->genes = array();
        $this->generation = 0;
    }

    /**
     * Get the best organisms from the given species.
     */
    protected function selectFromSpecies($species) {
        $count = count($species);
        $max = ceil($count * self::BEST_PERC);
        usort($species, function($a, $b) {
            return $b['score'] <=> $a['score'];
        });

        return array_slice($species, 0, $max);
    }

    /**
     * Spawn a neuron in the given network.
     */
    protected function addNode($network) {
        // Create a node.
        $layer = rand(1, count($network->getHiddenCounts()));
        $node = $network->addNode($layer);

        // We need to link this node for it to be effective..
        $start = $network->getRandomNode(range(0, $layer - 1));
        $end = $network->getRandomNode(range($layer + 1, count($network->getHiddenCounts()) + 1));
        $network->connect($start['layer'], $start['node'], $layer, $node);
        $network->connect($layer, $node, $end['layer'], $end['node']);
    }

    /**
     * Randomly mutate the given neural network.
     */
    protected function mutate($network) {
        $network = $network->clone();

        // Right. Lets do some things to this network.
        // We want to either link something or create a linked node.
        // Creating a linked node is likely to have a more dramatic impact on the network
        // so lets do that less often.
        // For the first three generations we have a much higher chance of creating nodes.
        $rand = rand(0, 100);

        if ($rand > 50) {
            // Link something.
            $start = $network->getRandomNode(range(0, count($network->getHiddenCounts())));
            $end = $network->getRandomNode(range($start['layer'] + 1, count($network->getHiddenCounts()) + 1));
            if ($start && $end) {
                $network->connect($start['layer'], $start['node'], $end['layer'], $end['node']);
            } else {
                $this->addNode($network);
            }

            // TODO - Make sure start is connected to something.
        } elseif ($rand > 10 && $this->generation > 2) {
            // Adjust the weight of something.
            // TODO - be nice if we could do this gradient-decent style.
            $newweight = mt_rand(0, 1000) / 1000;
            if (rand(0, 1) == 1) {
                $node = $network->getRandomNode(range(1, count($network->getHiddenCounts()) + 1));
                $network->setNodeWeight($node['layer'], $node['node'], $newweight);
            } else {
                $edge = $network->getRandomEdge();
                $network->setEdgeWeight($edge['layer'], $edge['node'], $edge['to'], $newweight);
            }
        } else {
            $this->addNode($network);
        }

        return $network;
    }

    /**
     * Store the given network on disk.
     */
    protected function store($species, $organism, $network) {
        $folder = dirname(__FILE__) . '/../../trained/game/snake/' . $this->generation . '/' . $species . '/';
        @mkdir($folder, 0777, true);
        $network->save($folder . '/' . $organism . '.nn');
    }

    /**
     * Create a new species out of a given organism.
     */
     protected function createSpecies($best, $scorefunc) {
        $species = array();
        for ($i = 0; $i < self::MAX_PER_SPECIES; $i++) {
            $network = $this->mutate($best);
            $this->store(count($this->species), $i, $network);
            $score = $scorefunc(function ($inputs) use ($network) {
                return $network->calculate($inputs);
            });
            $species[] = array('network' => $network, 'score' => $score);
        }
        $this->species = $species;

        return $species;
    }

    /**
     * Train the net.
     */
    public function run($scorefunc) {
        // Run the initial network and see what score we get.
        // We should never be worse than this.
        $network = $this->basenet->clone();
        $network->clear();
        $score = $scorefunc(function ($inputs) use ($network) {
            return $network->calculate($inputs);
        });
        $this->bestscore = $score;
        $this->bestnet = $network;

        for ($i = 0; $i < self::MAX_GENERATIONS; $i++) {
            $this->generation = $i;
            $prevspecies = $this->species;
            $this->species = array();

            // Generation 0 is a special case.
            if ($i == 0) {
                // Create the first species.
                $this->createSpecies($network, $scorefunc);
                continue;
            }

            // Select the best species from the previous generation and create a new species from them.
            $top = $this->selectFromSpecies($prevspecies);
            for ($j = 0; $j < self::MAX_SPECIES; $j++) {
                $organism = $top[array_rand($top)];
                $this->createSpecies($organism['network'], $scorefunc);
            }

            // Right. Choose the best of this generation.
            foreach ($this->species[$i] as $species => $organism) {
                if ($organism['score'] > $this->bestscore) {
                    echo "Generation {$i}, Species {$species} produced a better result.<br />";
                    $this->bestscore = $organism['score'];
                    $this->bestnet = $organism['network'];
                }
            }
        }
    }

    /**
     * Return the best net.
     */
    public function bestNetwork() {
        return $this->bestnet;
    }
}
