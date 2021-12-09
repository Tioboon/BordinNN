package BordinNN;

import java.util.ArrayList;
import java.util.List;

public class HiddenLayers {
    public List<LayerDense> layerList;

    public HiddenLayers(List<LayerDense> layerList){
        this.layerList = layerList;
    }

    public HiddenLayers Copy(){
        List<LayerDense> copyLayerList = new ArrayList<>();
        for (LayerDense layerDense: layerList) {
            copyLayerList.add(layerDense.Copy());
        }
        return new HiddenLayers(copyLayerList);
    }

}
