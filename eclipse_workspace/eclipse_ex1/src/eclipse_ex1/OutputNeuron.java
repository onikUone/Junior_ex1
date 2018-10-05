package eclipse_ex1;

import java.math.BigDecimal;
import java.util.Arrays;

public class OutputNeuron {
	//member field
	BigDecimal[] weight;
	BigDecimal threshoud;
	BigDecimal net;
	BigDecimal eta;
	BigDecimal alpha;
	BigDecimal[] delta_W;
	BigDecimal delta_T;

	//member method
	public BigDecimal[] getWeight(){
		return weight;
	}
	public void clearNet(){
		net = new BigDecimal("0.0");
	}
	public void setNet(BigDecimal o_fromInter, int interNumber){
		net = net.add(weight[interNumber].multiply(o_fromInter));
	}
	public void calcNet(){
		net = net.add(threshoud);
	}
	public BigDecimal output(){
		return sigmoid(net);
	}
	public static BigDecimal sigmoid(BigDecimal net) {
		double s = 1 / (1 + Math.exp((-1.0) * Double.parseDouble(String.valueOf(net))));
		return BigDecimal.valueOf(s);
	}
	public void reWeight(BigDecimal y, BigDecimal o_fromOutput, BigDecimal o_fromInter[]){
		for(int i=0; i<weight.length; i++){
			delta_W[i] = delta_W[i].multiply(alpha);
			delta_W[i] = delta_W[i].add(eta.multiply((y.subtract(o_fromOutput)).multiply(o_fromOutput.multiply((new BigDecimal("1.0").subtract(o_fromOutput)).multiply(o_fromInter[i])))));
			weight[i] = weight[i].add(delta_W[i]);	//重み更新式
		}
		delta_T = delta_T.multiply(alpha);
		delta_T = delta_T.add(eta.multiply((y.subtract(o_fromOutput)).multiply(o_fromOutput.multiply((new BigDecimal("1.0").subtract(o_fromOutput))))));
		threshoud = threshoud.add(delta_T);	//しきい値更新式
	}

	//constructor
	OutputNeuron(int interNumber){
		weight = new BigDecimal[interNumber];
		for(int i=0; i<interNumber; i++){
			weight[i] = new BigDecimal("0.5");
		}
		threshoud = new BigDecimal("0.5");
		eta = new BigDecimal("0.5");
		alpha = new BigDecimal("0.9");
		delta_W = new BigDecimal[weight.length];
		Arrays.fill(delta_W, new BigDecimal("0.0"));
		delta_T = new BigDecimal("0.0");
	}
}
