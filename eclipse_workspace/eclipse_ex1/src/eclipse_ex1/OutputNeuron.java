package eclipse_ex1;

import java.util.Arrays;

public class OutputNeuron {
	//member field
	double[] weight;
	double threshoud;
	double net;
	double eta;
	double alpha;
	double[] delta_W;
	double delta_T;

	//member method
	public double[] getWeight(){
		return weight;
	}
	public void clearNet(){
		net = 0.0;
	}
	public void setNet(double o_fromInter, int interNumber){
		net += weight[interNumber] * o_fromInter;
	}
	public void calcNet(){
		net += threshoud;
	}
	public double output(){
		return 1.0/(1.0 + Math.exp(-net));
	}

	public void reWeight(double y, double o_fromOutput, InterNeuron inter[]){
		for(int i=0; i<weight.length; i++){
			delta_W[i] *= alpha;
			delta_W[i] += eta * (y - o_fromOutput) * o_fromOutput * (1 - o_fromOutput) * inter[i].output();
			weight[i] += delta_W[i];	//重み更新式
		}
		delta_T *= alpha;
		delta_T += eta * (y - o_fromOutput) * o_fromOutput * (1 - o_fromOutput);
		threshoud += delta_T;	//しきい値更新式
	}

	//constructor
	OutputNeuron(int interNumber){
		weight = new double[interNumber];
		for(int i=0; i<interNumber; i++){
			weight[i] = 0.5;
		}
		threshoud = 0.5;
		eta = 0.5;
		alpha = 0.9;
		delta_W = new double[weight.length];
		Arrays.fill(delta_W, 0.0);
		delta_T = 0.0;
	}
}
