// Read-only reference linkage: original BPA/DS .cpp files are compiled unchanged.
#include <iostream>
#include <iomanip>
#include <memory>
#include "vision_bpa.h"
#include "temp_bpa.h"
#include "uv_bpa.h"
#include "ds_fusion.h"
int main(){
 std::unique_ptr<VisionBPA> v; std::unique_ptr<ThermalBPA> t; std::unique_ptr<UVBPA> u;
 DS_Fusion ds; int reset,uv; double rc,tc,temp;
 std::cout<<std::setprecision(17);
 while(std::cin>>reset>>rc>>tc>>temp>>uv){
  if(reset || !v){v.reset(new VisionBPA(10,.7));t.reset(new ThermalBPA(10,.7,50));u.reset(new UVBPA(.7,.01,.1));}
  BPAResult a=v->update(rc,.3), b=t->update(tc,temp), c=u->update(uv);
  BPAResult legacy=ds.fuse3(a,b,c), classic=ds.fuse(ds.fuse(a,b),c);
  for(auto m:{a,b,c,legacy,classic}) std::cout<<m.fire<<" "<<m.nofire<<" "<<m.uncertain<<" ";
  std::cout<<"\n";
 }
 return 0;
}
