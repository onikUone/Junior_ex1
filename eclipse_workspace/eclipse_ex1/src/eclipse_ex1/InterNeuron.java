package eclipse_ex1;

import java.util.Arrays;

public class InterNeuron {
	//member field
	double[] weight;
	double threshoud;
	double net = 0;
	double eta;
	double alpha;
	double[] delta_W;
	double delta_T;

	//member method
	public void clearNet(){
		net = 0;
	}
	public void setNet(double o_fromInput){
		net += weight[0] * o_fromInput;
	}
	public void calcNet(){
		net += threshoud;
	}
	public double output() {
		return sigmoid(net);
	}
	public static double sigmoid(double net) {
		return 1 / (1 + Math.exp((-1) * net));
	}
	public void reWeight(double x, double y, double o_fromInter, double o_fromOutput, OutputNeuron out){
		double temp = 0;
		for(int i=0; i<out.getWeight().length; i++){
			temp += (y - o_fromOutput) * o_fromOutput * (1 - o_fromOutput) * out.getWeight()[i];
		}
		for(int i=0; i<weight.length; i++){
			delta_W[i] *= alpha;
			delta_W[i] += eta * o_fromInter * (1 - o_fromInter) * temp * x;
			weight[i] += delta_W[i];
		}
		delta_T *= alpha;
		delta_T += eta * o_fromInter * (1 - o_fromInter) * temp;
		threshoud += delta_T;
	}
	//constructor
	//引数に入力層の個数を指定すると、その個数だけの次元で結合強度配列を作成する。
	InterNeuron(int inputNumber) {
		weight = new double[inputNumber];
		for(int i=0; i<inputNumber; i++){
			weight[i] = 0.5;
		}
		threshoud = 0.5;
		eta = 0.5;
		alpha = 0.9;
		delta_W = new double[weight.length];
		Arrays.fill(delta_W, 0);
		delta_T = 0;
	}
}
