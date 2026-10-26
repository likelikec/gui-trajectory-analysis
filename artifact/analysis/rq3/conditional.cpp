#include <vector>
#include <cmath>
#include <algorithm>
// Exact conditional likelihood, gradient and observed information.
// Complement the case set for numerical stability; no likelihood approximation.
extern "C" void evaluate(int G,int M,int P,const double* X,const double* Y,
 const long long* sizes,const double* beta,const double* weights,double* out){
 std::fill(out,out+1+P+P*P,0.0);
 for(int g=0;g<G;++g){
  if(weights[g]==0)continue;
  int n=sizes[g],s=0;for(int i=0;i<n;++i)s+=(int)Y[g*M+i];
  bool comp=s>n-s;int m=std::min(s,n-s);double sign=comp?-1:1;
  std::vector<double>D(m+1,-INFINITY),Q((m+1)*P,0),H((m+1)*P*P,0),z(P),right(P),delta(P);
  D[0]=0;double num=0;std::vector<double>sn(P,0);
  for(int i=0;i<n;++i){
   double eta=0;for(int a=0;a<P;++a){z[a]=sign*X[(g*M+i)*P+a];eta+=z[a]*beta[a];}
   double y=comp?1-Y[g*M+i]:Y[g*M+i];num+=y*eta;for(int a=0;a<P;++a)sn[a]+=y*z[a];
   for(int j=std::min(i+1,m);j>=1;--j){
    double aa=D[j],bb=D[j-1]+eta,top=std::max(aa,bb),den=top+std::log(std::exp(aa-top)+std::exp(bb-top));
    double wa=std::exp(aa-den),wb=std::exp(bb-den);
    for(int a=0;a<P;++a){right[a]=Q[(j-1)*P+a]+z[a];delta[a]=Q[j*P+a]-right[a];}
    for(int a=0;a<P;++a)for(int b=0;b<P;++b){int u=j*P*P+a*P+b,v=(j-1)*P*P+a*P+b;H[u]=wa*H[u]+wb*H[v]+wa*wb*delta[a]*delta[b];}
    for(int a=0;a<P;++a)Q[j*P+a]=wa*Q[j*P+a]+wb*right[a];D[j]=den;
   }
  }
  out[0]+=weights[g]*(num-D[m]);
  for(int a=0;a<P;++a){out[1+a]+=weights[g]*(sn[a]-Q[m*P+a]);for(int b=0;b<P;++b)out[1+P+a*P+b]+=weights[g]*H[m*P*P+a*P+b];}
 }
}
