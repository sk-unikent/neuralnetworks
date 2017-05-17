<?php

namespace KeltyNN\Trainers;

interface Trainer
{
    function run($scorefunc);
    function bestNetwork();
}
