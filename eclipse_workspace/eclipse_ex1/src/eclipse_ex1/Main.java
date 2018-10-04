package eclipse_ex1;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class Main {

	public static double[][] inputFile(String path) throws IOException{
		List<String[]> list = new ArrayList<String[]>();
		BufferedReader in = new BufferedReader(new FileReader(path));
		String line;
		while((line = in.readLine()) != null){
//			System.out.println(line);	//一行ずつprintできる
			list.add(line.split("\t"));
		}
		in.close();
		double[][] x = new double[list.size()][list.get(0).length];
		for(int i=0; i<list.size(); i++){
			x[i][0] = Double.parseDouble(list.get(i)[0]);
			x[i][1] = Double.parseDouble(list.get(i)[1]);
		}

		for(int i=0; i<x.length; i++){
			System.out.println(x[i][0] + " " + x[i][1]);
		}

		return x;
	}

	public static void forward(double x[], InterNeuron inter[], OutputNeuron out){
		for(int i=0; i<inter.length; i++){
			inter[i].clearNet();
		}
		out.clearNet();
		for(int i=0; i<inter.length; i++){
			inter[i].setNet(x[0]);
		}
		for(int i=0; i<inter.length; i++){
			inter[i].calcNet();
		}
		for(int i=0; i<inter.length; i++){
			out.setNet(inter[i].output(), i);
		}
		out.calcNet();
//		System.out.println(out.output());
	}

	public static void main(String[] args) throws IOException {
		//ファイル読み込み
		double[][] inputFile;	//datファイル ２次元配列化
		String path = "/Users/Uone/IDrive/OPU/研究フォルダ/1_プログラミング課題/eclipse_workspace/eclipse_ex1/src/eclipse_ex1/inputData.dat";	//Mac(ノートPC)環境
//		String path = "C:\\Users\\Yuichi Omozaki\\IDrive\\Junior_ex1\\eclipse_workspace\\eclipse_ex1\\src\\eclipse_ex1/inputData.dat";	//Windows(研究室環境)
		inputFile = inputFile(path);

		//教師データ作成
		double[] x = new double[inputFile.length];	//入力ベクトル_教師データ
		double[] y = new double[inputFile.length];	//出力ベクトル_教師データ
		for(int i=0; i<inputFile.length; i++){
			x[i] = inputFile[i][0];
			y[i] = inputFile[i][1];
		}
		System.out.println("main");

		//ニューロン実装
		InterNeuron inter[] = new InterNeuron[20];	//中間層ニューロンの個数はここで指定する
		for(int i=0; i<inter.length; i++){
			inter[i] = new InterNeuron(1);
		}
		OutputNeuron out = new OutputNeuron(inter.length);	//inter.lengthで中間層ニューロンの個数指定し、その数だけ結合強度を用意する。

		//順方向計算
		double o_fromOutput;
		double o_fromInter[] = new double[inter.length];
		forward(x, inter, out);
		for(int i=0; i<inter.length; i++){
			o_fromInter[i] = inter[i].output();
		}
		o_fromOutput = out.output();





	}

}
