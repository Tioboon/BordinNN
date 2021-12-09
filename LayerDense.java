package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class LayerDense {
    //Bunch of neurons that resides in this hidden layer
    List<Neuron> neurons;

    public VectorResultNeuron actualNeuronVector;

    public VectorResultNeuron nextNeuronVector;

    public void GetOutPutByBatch(InputBatch inputBatch){
        List<InputForNeurons> inputForNeurons = new ArrayList<>();
        for (InputForNeurons input: inputBatch.inputBatchList) {
            List<Double> neuronOutPut = new ArrayList<>();
            for (Neuron neuron: neurons) {
                neuronOutPut.add(neuron.GetResultByInput(input.inputList));
            }
            inputForNeurons.add(new InputForNeurons(neuronOutPut));
        }
        InputBatch output = new InputBatch(inputForNeurons);
        VectorResultNeuron actualNeuronVector = new VectorResultNeuron(output);
        this.actualNeuronVector = actualNeuronVector;
    }

    public void GetFutureOutPut(InputBatch futureBatch){
        List<InputForNeurons> nextInputList = new ArrayList<>();
        for (InputForNeurons input: futureBatch.inputBatchList) {
            List<Double> neuronOutPut = new ArrayList<>();
            for (Neuron neuron: neurons) {
                neuronOutPut.add(neuron.GetResultByInput(input.inputList));
            }
            nextInputList.add(new InputForNeurons(neuronOutPut));
        }
        InputBatch nextOutPut = new InputBatch(nextInputList);
        VectorResultNeuron nextResultNeuron = new VectorResultNeuron(nextOutPut);
        this.nextNeuronVector = nextResultNeuron;
    }

    public LayerDense Copy(){
        List<Neuron> copiedNeurons = new ArrayList<>();
        for (Neuron neuron: neurons) {
            copiedNeurons.add(neuron.Copy());
        }
        return new LayerDense(copiedNeurons);
    }

    //Don't work very well
    public void TweakValues(Slope slope) {
        for(int i = 0; i < slope.derivativeOutPut.inputBatchList.size(); i++){
            for (int j = 0; j < slope.derivativeOutPut.inputBatchList.get(0).inputList.size(); j++) {
                for(int k = 0; k < neurons.get(j).weights.size(); k++){
                    neurons.get(j).weights.set(k,
                            neurons.get(j).weights.get(k) +
                            .1d * MathHelper.GetRandomPositiveOrNegative(
                                    slope.derivativeOutPut.inputBatchList.get(i).inputList.get(j)
                            )
                    );
                }
            }
        }

        for(int i = 0; i < slope.derivativeOutPut.inputBatchList.size(); i++){
            for (int j = 0; j < slope.derivativeOutPut.inputBatchList.get(0).inputList.size(); j++) {
                    neurons.get(j).bias = neurons.get(j).bias +
                            .01d * MathHelper.GetRandomPositiveOrNegative(
                                    slope.bVariableOutPut.inputBatchList.get(i).inputList.get(j)
                            );
            }
        }
    }


    public LayerDense(List<Neuron> neurons){
        this.neurons = neurons;
    }

    public LayerDense(List<InputForNeurons> input, int nOfNeurons){
        neurons = new ArrayList<>(nOfNeurons);
        for (int i = 0; i < nOfNeurons; i++){
            neurons.add(new Neuron(input.size()));
        }
    }
}
