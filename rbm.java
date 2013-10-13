//Code written by Gengtao Jia 
//This program trains Restricted Boltzmann Machine in which
// visible, binary, stochastic pixels are connected to hidden,
// binary, stochastic feature detectors using symmetrically weighted
// connections. Learning is done with 1-step Contrastive Divergence.   
// The program assumes that the following variables are set externally:
// maxepoch  -- maximum number of epochs
// numhid    -- number of hidden units 
// batchdata -- the data that is divided into batches (numcases numdims numbatches)
// restart   -- set to 1 if learning starts from beginning 

//import Jama.Matrix;
import java.util.Random;
import java.lang.Math;

public class rbm {
	private double epsilonw;
	private double epsilonvb;
	private double espilonhb;
	private double weightcost;
	private double initialmomentum;
	private double finalmomentum;
	private int numhid;
	private int numdims;
	private int numbatches;
	private int maxepoch;
	
	private Matrix hidbiases;
	private Matrix visbiases;
	private Matrix poshidprobs;
	private Matrix neghidprobs;
	private Matrix posprods;
	private Matrix negprods;
	private Matrix vishidinc;
	private Matrix hdibiasinc;
	private Matrix visbiasinc;
	private Matrix data;
	private Matrix vishid;
	
    //generate the initial random weights of each path	
    
	private void  initialize(Matrix data, int numdims, int numhid){
		epsilonw = 0.1; 
		epsilonvb = 0.1;
		espilonhb = 0.1;
		weightcost = 0.0002;
		initialmomentum = 0.5;
		finalmomentum = 0.9;
		
		this.numhid = numhid;
		this.numdims = numdims;
		
		this.data = data.copy();
		hidbiases = new Matrix(1,numhid);
		visbiases = new Matrix(1,numdims);
		poshidprobs = new Matrix(1,numhid);
		neghidprobs = new Matrix(1,numhid);
		posprods = new Matrix(numdims,numhid);
		negprods = new Matrix(numdims,numhid);
		vishidinc = new Matrix(numdims,numhid);
		hdibiasinc = new Matrix(1,numhid);
		visbiasinc = new Matrix(1,numhid);
		
		double[][] temp1 = new double[numdims][numhid];
		Random randomgenerator = new Random();
		for(int i = 0; i < numdims; i++){
			for(int j = 0; j < numhid; j++){
				temp1[i][j] = 0.1 * randomgenerator.nextGaussian();
				}
			}
		vishid = new Matrix(temp1); //random Matrix(numdims,numhid);
	}
	//Start calculate the positive phase
	//calculate the cured value of h0
	
	private Matrix getposphase(){
		poshidprobs = data.times(vishid);
		//(1 * numdims) * (numdims * numhid)
		poshidprobs.plusEquals(hidbiases);
		//data*vishid + hidbiases
		double [] [] product_tmp2 = poshidprobs.getArray();
		int i2 = 0;
		while( i2 < numhid){
			product_tmp2[0][i2] = 1/(1 + Math.exp(-product_tmp2[0][i2]));
			i2++;
		}
		posprods = data.transpose().times(poshidprobs);
		
	//end of the positive phase calculation, find the binary presentation of h0
		int i3 =0;
		double [] [] tmp1 = poshidprobs.getArray();
		double [] [] tmp2 = new double [1][numhid];
		Random randomgenerator = new Random();
		while (i3 < numhid){
			if (tmp1[0][i3] > randomgenerator.nextDouble())
				tmp2[0][i3] = 1;
			else tmp2[0][i3] = 0;
			i3++;
			}
		Matrix poshidstates = new Matrix(tmp2);
		return (poshidstates);
		}
	
	//start calculate the negative phase
	//calculate the curved value of v1,h1
	private void getnegphase(Matrix poshidstates){
		//find the vector of v1
		Matrix negdata = poshidstates.times(vishid.transpose());
		//(1 * numhid) * (numhid * numdims) = (1 * numdims)
		negdata.plusEquals(visbiases);
		//poshidstates*vishid' + visbiases
		double [] [] tmp1 = negdata.getArray();
		int i1 = 0;
		while( i1 < numdims){
			tmp1[0][i1] = 1/(1 + Math.exp(-tmp1[0][i1]));
			i1++;
		}
		
		//find the vector of h1
		neghidprobs = negdata.times(vishid);
		//(1 * numdims) * (numdims * numhid) = (1 * numhid)
		neghidprobs.plusEquals(hidbiases);
		double [] [] tmp2 = neghidprobs.getArray();
		int i2 = 0;
		while( i2 < numhid){
			tmp2[0][i2] = 1/(1 + Math.exp(-tmp2[0][i2]));
			i2++;
			}
		negprods = negdata.transpose().times(neghidprobs);
		//(numdims * 1) *(1 * numhid) = (numdims * numhid)
		}
	
	//update the weights and biases
	private void update(int epoch){
		double momentum;
		if (epoch > 5)
			momentum = finalmomentum;
		else
			momentum = initialmomentum;
		//vishidinc = momentum*vishidinc + epsilonw*( (posprods-negprods)/numcases - weightcost*vishid);
		vishidinc.timesEquals(momentum);
	    Matrix temp1 = posprods.minus(negprods);
	    Matrix temp2 = vishid.times(weightcost);
	    temp1.minusEquals(temp2);
	    temp1.timesEquals(epsilonw);
	    vishidinc.plusEquals(temp1);
	    //vishid = vishid + vishidinc;
	    vishid.plusEquals(vishidinc);
	    }
	
	public int run(int epochmax, Matrix inputData, int numdims, int numhid){
		initialize(inputData, numdims, numhid);
		for (int i = 0; i < epochmax; i++ ){
			System.out.print("Epoch: ");
			System.out.println(i);
			Matrix poshidstates = getposphase();
			getnegphase(poshidstates);
			update(i);
		}
		return (0);
		}
	}
	
