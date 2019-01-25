//This code shows how to define neuron activations and predefined netlets.
#include <cassert>
#include <vector>
#include <iostream>
using namespace std;

typedef struct {
  inline const char *name()  { return "Elliott";}
  inline int G(int x) {return -1.f*x; }
  inline float G(float x) {return 2.f*x; }
  inline double G(double x) {return 4*x; }
} Elliot_nn;

typedef struct {
  inline const char *name()  { return "Tanh";}
  inline int G(int x) {return -1.f*x; }
  inline float G(float x) {return 2.f*x; }
  inline double G(double x) {return 4*x; }
} Tanh_nn;

typedef struct {
  inline const char *name()  { return "netlet";}
  inline int G(vector<int> &&v) {
    assert(v.size()==2);
    return -1.f *(v[0]+v[1]);
  }
} Netlet_nn;

int main()
{
  Elliot_nn x;

  cout << x.name() << " is an integer " << x.G(1) << " size " << sizeof(x) << endl;
  cout << x.name() << " is a float " << x.G(1.f) << " size " << sizeof(x) << endl;
  cout << x.name() << " is a double " << x.G(1.) << " size " << sizeof(x) << endl;

  Tanh_nn a[100000]; 

  cout << sizeof(a) << endl;
  cout << a[0].name() << " " << a[0].G(1) << " size " << sizeof(a[0]) << endl;

  Netlet_nn y;
  vector<int> v={2,3};
  // these avoid copy operations
  cout << y.name() << " " << y.G(move(v)) << endl;
  cout << y.name() << " " << y.G(vector<int>{10,20}) << endl;
  cout << y.name() << " " << y.G(vector<int>{10,x.G(1)}) << endl;
  // this will fail if compiled with asserts on
  //cout << y.name() << " " << y.G(vector<int>{10,x.G(1),1}) << endl;
}
