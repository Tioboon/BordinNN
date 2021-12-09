package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class VectorResultNeuron {
    InputBatch xOutPut;
    InputBatch yOutPut;

    public VectorResultNeuron(InputBatch outPut){
        this.xOutPut = outPut;
        yOutPut = CalculateY(outPut);
    }

    private InputBatch CalculateY(InputBatch normalInput){
        List<InputForNeurons> yValuesInput = new ArrayList<>();
        for (InputForNeurons input : normalInput.inputBatchList) {
            List<Double> yValuesList = new ArrayList<>();
            for (double values: input.inputList) {
                yValuesList.add(MathHelper.NonLinearF(values));
            }
            yValuesInput.add(new InputForNeurons(yValuesList));
        }
        return new InputBatch(yValuesInput);
    }
}
