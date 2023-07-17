


from DeepJetCore.TrainData import TrainData, fileTimeOut
from DeepJetCore import SimpleArray
import numpy as np
import uproot3 as uproot
import awkward as ak1
from numba import jit
import gzip
import os
import pickle
import pandas as pd

n_id_classes = 22

def calc_eta(x, y, z):
    rsq = np.sqrt(x ** 2 + y ** 2)
    return -1 * np.sign(z) * np.log(rsq / np.abs(z + 1e-3) / 2.+1e-3)

def calc_phi(x, y, z):
    return np.arctan2(y,x)#cms like

# One hot encodings of the particles

# particles with freq less than 1000
# other = [310.0, 1000070144.0, 1000120256.0, 1000200448.0, 221.0, -12.0, 1000080192.0, 1000140288.0, 1000210496.0, 1000220480.0, -2112.0, -321.0, 1000230464.0, 1000130240.0, 1000260544.0, 1000110272.0, 1000250560.0, 1000240512.0, 1000030080.0, 1000110208.0, 1000060160.0, 321.0, 1000180352.0, 1000030016.0, 331.0, 1000250496.0, 3222.0, 3112.0, 3212.0, 3122.0, 113.0, 1000040128.0, 1000190400.0, 1000160320.0, 1000210432.0, 1000230528.0, 1000200384.0, 1000180416.0, 1000260480.0, 1000100224.0, 1000130304.0, 1000120192.0, 1000020096.0, 1000090176.0, 223.0, 1000220416.0, -3122.0, 1000170368.0, 1000090240.0, 1000100160.0, 1000190464.0, 1000050048.0, 1000150336.0, -3212.0, -411.0, 4122.0, 1000140224.0, 1000280576.0]

particle_ids = [-2212.0, -211.0, -14.0, -13.0, -11.0, 11.0, 12.0, 13.0, 14.0, 22.0, 111.0, 130.0, 211.0, 2112.0, 2212.0, 1000010048.0, 1000020032.0, 1000040064.0, 1000050112.0, 1000060096.0, 1000080128.0]
# IMPORTANT: use absolute_value and sign in a separate field

particle_ids = [int(x) for x in particle_ids]
#other = [int(x) for x in other]

#@jit(nopython=False)
def truth_loop(link_list :list, 
               t_dict:dict,
               part_p_list :list,
               part_pid_list: list,
               part_theta_list: list,
               part_phi_list: list,
               ):
    
    nevts = len(link_list)
    for ie in range(nevts):#event
        nhits  = len(link_list[ie])
        for ih in range(nhits):
            idx = -1
            mom = 0.
            t_pos = [0.,0.,0.]
            t_pid = [0.] * (len(particle_ids) + 1) # "other" category
            assert len(t_pid) == len(particle_ids) + 1
            # 0th entry for the sign, 1st entry for "OTHER"
            if link_list[ie][ih] >= 0:
                idx = link_list[ie][ih]
                mom = part_p_list[ie][idx]
                particle_id = 0
                if (part_pid_list[ie][idx]) in particle_ids:
                    particle_id = particle_ids.index((part_pid_list[ie][idx])) + 1
                # t_pid[0] = np.sign(part_pid_list[ie][idx]) # don't encode separate sign...
                t_pid[int(particle_id)] = 1.
                part_theta, part_phi = part_theta_list[ie][idx], part_phi_list[ie][idx]
                r = mom
                x_part = r * np.sin(part_theta) * np.cos(part_phi)
                y_part = r * np.sin(part_theta) * np.sin(part_phi)
                z_part = r * np.cos(part_theta)
                t_pos = [x_part, y_part, z_part]

                
            t_dict['t_idx'].append([idx])
            t_dict['t_energy'].append([mom])
            t_dict['t_pos'].append(t_pos)
            t_dict['t_time'].append([0.])
            t_dict['t_pid'].append(t_pid)
            t_dict['t_spectator'].append([0.])
            t_dict['t_fully_contained'].append([1.])
            t_dict['t_rec_energy'].append([mom]) # THIS WILL NEED TO BE ADJUSTED
            t_dict['t_is_unique'].append([1]) #does not matter really
    
    
    return t_dict
    

class TrainData_fcc(TrainData):
   
    def branchToFlatArray(self, b, return_row_splits=False, dtype='float32'):
        
        a = b.array()
        nevents = a.shape[0]
        rowsplits = [0]
        
        for i in range(nevents):
            rowsplits.append(rowsplits[-1] + a[i].shape[0])
        
        rowsplits = np.array(rowsplits, dtype='int64')
        
        if return_row_splits:
            return np.expand_dims(np.array(a.flatten(),dtype=dtype), axis=1),np.array(rowsplits, dtype='int64') 
        else:
            return np.expand_dims(np.array(a.flatten(),dtype=dtype), axis=1)
    def interpretAllModelInputs(self, ilist, returndict=True):
        if not returndict:
            raise ValueError('interpretAllModelInputs: Non-dict output is DEPRECATED. PLEASE REMOVE') 
        '''
        input: the full list of keras inputs
        returns: td
         - rechit feature array
         - t_idx
         - t_energy
         - t_pos
         - t_time
         - t_pid :             non hot-encoded pid
         - t_spectator :       spectator score, higher: further from shower core
         - t_fully_contained : fully contained in calorimeter, no 'scraping'
         - t_rec_energy :      the truth-associated deposited 
                               (and rechit calibrated) energy, including fractional assignments)
         - t_is_unique :       an index that is 1 for exactly one hit per truth shower
         - row_splits
         
        '''
        out = {
            'features':ilist[0],
            'rechit_energy': ilist[0][:,0:1], #this is hacky. FIXME
            't_idx':ilist[2],
            't_energy':ilist[4],
            't_pos':ilist[6],
            't_time':ilist[8],
            't_pid':ilist[10],
            't_spectator':ilist[12],
            't_fully_contained':ilist[14],
            'row_splits':ilist[1]
            }
        #keep length check for compatibility
        if len(ilist)>16:
            out['t_rec_energy'] = ilist[16]
        if len(ilist)>18:
            out['t_is_unique'] = ilist[18]
        return out
    
    def createPandasDataFrame(self, eventno=-1):
        #since this is only needed occationally
        
        if self.nElements() <= eventno:
            raise IndexError("Event wrongly selected")
        
        tdc = self.copy()
        if eventno>=0:
            tdc.skim(eventno)
        
        f = tdc.transferFeatureListToNumpy(False)
        featd = self.createFeatureDict(f[0])
        rs = f[1]
        truthd = self.createTruthDict(f)
        
        featd.update(truthd)
        
        del featd['recHitXY'] #so that it's flat
        
        featd['recHitLogEnergy'] = np.log(featd['recHitEnergy']+1.+1e-8)
        
        #allarr = []
        #for k in featd:
        #    allarr.append(featd[k])
        #allarr = np.concatenate(allarr,axis=1)
        #
        #frame = pd.DataFrame (allarr, columns = [k for k in featd])
        #for k in featd.keys():
        #    featd[k] = [featd[k]]
        #frame = pd.DataFrame()
        for k in featd.keys():
            #frame.insert(0,k,featd[k])
            if featd[k].shape[1] == 1:
                featd[k] = np.squeeze(featd[k],axis=1)
            elif k=='truthHitAssignedPIDs' or k== 't_pid':
                featd[k] =  np.argmax(featd[k], axis=-1)
            else:
                raise ValueError("only pid one-hot allowed to have more than one additional dimension, tried to squeeze "+ k)
        
        frame = pd.DataFrame.from_records(featd)
        
        if eventno>=0:
            return frame
        else:
            return frame, rs

    def createFeatureDict(self,infeat,addxycomb=True):
        '''
        infeat is the full list of features, including truth
        '''
        
        #small compatibility layer with old usage.
        feat = infeat
        if type(infeat) == list:
            feat=infeat[0]
        
        d = {
        'recHitEnergy': feat[:,0:1] ,          #recHitEnergy,
        'recHitEta'   : feat[:,1:2] ,          #recHitEta   ,
        'recHitID'    : feat[:,2:3] ,          #recHitID, #indicator if it is track or not
        'recHitTheta' : feat[:,3:4] ,          #recHitTheta ,
        'recHitR'     : feat[:,4:5] ,          #recHitR   ,
        'recHitX'     : feat[:,5:6] ,          #recHitX     ,
        'recHitY'     : feat[:,6:7] ,          #recHitY     ,
        'recHitZ'     : feat[:,7:8] ,          #recHitZ     ,
        'recHitTime'  : feat[:,8:9] ,            #recHitTime  
        'recHitHitR'  : feat[:,9:10] ,            #recHitTime  
        }
        if addxycomb:
            d['recHitXY']  = feat[:,5:7]    
            
        return d
    
    def     reDict(self,infeat,addxycomb=True):
            '''
            infeat is the full list of features, including truth
            '''
            
            #small compatibility layer with old usage.
            feat = infeat
            if type(infeat) == list:
                feat=infeat[0]
            
            d = {
            'recHitEnergy': feat[:,0:1] ,          #recHitEnergy,
            'recHitEta'   : feat[:,1:2] ,          #recHitEta   ,
            'recHitID'    : feat[:,2:3] ,          #recHitID, #indicator if it is track or not
            'recHitTheta' : feat[:,3:4] ,          #recHitTheta ,
            'recHitR'     : feat[:,4:5] ,          #recHitR   ,
            'recHitX'     : feat[:,5:6] ,          #recHitX     ,
            'recHitY'     : feat[:,6:7] ,          #recHitY     ,
            'recHitZ'     : feat[:,7:8] ,          #recHitZ     ,
            'recHitTime'  : feat[:,8:9] ,            #recHitTime  
            'recHitHitR'  : feat[:,9:10] ,            #recHitTime  
            }
            if addxycomb:
                d['recHitXY']  = feat[:,5:7]    
                
            return d
  
    def createTruthDict(self, allfeat, truthidx=None):
        '''
        This is deprecated and should be replaced by a more transparent way.
        '''
        #print(__name__,'createTruthDict: should be deprecated soon and replaced by a more uniform interface')
        data = self.interpretAllModelInputs(allfeat,returndict=True)
        
        out={
            'truthHitAssignementIdx': data['t_idx'],
            'truthHitAssignedEnergies': data['t_energy'],
            'truthHitAssignedX': data['t_pos'][:,0:1],
            'truthHitAssignedY': data['t_pos'][:,1:2],
            'truthHitAssignedZ': data['t_pos'][:,2:3],
            'truthHitAssignedEta': calc_eta(data['t_pos'][:,0:1], data['t_pos'][:,1:2], data['t_pos'][:,2:3]),
            'truthHitAssignedPhi': calc_phi(data['t_pos'][:,0:1], data['t_pos'][:,1:2], data['t_pos'][:,2:3]),
            'truthHitAssignedT': data['t_time'],
            'truthHitAssignedPIDs': data['t_pid'],
            'truthHitSpectatorFlag': data['t_spectator'],
            'truthHitFullyContainedFlag': data['t_fully_contained'],
            }
        if 't_rec_energy' in data.keys():
            out['t_rec_energy']=data['t_rec_energy']
        if 't_hit_unique' in data.keys():
            out['t_is_unique']=data['t_hit_unique']
        return out
    

    def convertFromSourceFile(self, filename, weighterobjects, istraining, treename="events"):
        
        fileTimeOut(filename, 10)#wait 10 seconds for file in case there are hiccups
        tree = uproot.open(filename)[treename]
        
        '''
        
        hit_x, hit_y, hit_z: the spatial coordinates of the voxel centroids that registered the hit
        hit_dE: the energy registered in the voxel (signal + BIB noise)
        recHit_dE: the 'reconstructed' hit energy, i.e. the energy deposited by signal only
        evt_dE: the total energy deposited by the signal photon in the calorimeter
        evt_ID: an int label for each event -only for bookkeeping, should not be needed
        isSignal: a flag, -1 if only BIB noise, 0 if there is also signal hit deposition

        '''
        
        hit_x, rs = self.branchToFlatArray(tree["hit_x"], True)
        hit_y = self.branchToFlatArray(tree["hit_y"])
        hit_z = self.branchToFlatArray(tree["hit_z"])
        hit_t = self.branchToFlatArray(tree["hit_t"])
        hit_e = self.branchToFlatArray(tree["hit_e"])
        hit_theta = self.branchToFlatArray(tree["hit_theta"])
        #hit_type = self.branchToFlatArray(tree["hit_type"])
        
        zerosf = 0.*hit_e
        
        print('hit_e',hit_e)
        hit_e = np.where(hit_e<0., 0., hit_e)
        
        
        farr = SimpleArray(np.concatenate([
            hit_e,
            zerosf,
            zerosf, #indicator if it is track or not
            zerosf,
            hit_theta,
            hit_x,
            hit_y,
            hit_z,
            zerosf,
            hit_t
            ], axis=-1), rs,name="recHitFeatures") # TODO: add hit_type


        # create truth
        hit_genlink = tree["hit_genlink0"].array()
        part_p = tree["part_p"].array()
        
        t = {
            't_idx' : [], #names are optional
            't_energy' :  [],
            't_pos' :  [], #three coordinates
            't_time' : []  ,
            't_pid' :  [] , #6 truth classes
            't_spectator' :  [],
            't_fully_contained' :  [],
            't_rec_energy' :  [],
            't_is_unique' :  []
            }
        
        #do this with numba
        # print("Part pids", tree["part_pid"].array().tolist())
        t = truth_loop(hit_genlink.tolist(), 
                       t,
                       part_p.tolist(),
                       tree["part_pid"].array().tolist(),
                       tree["part_theta"].array().tolist(),
                       tree["part_phi"].array().tolist(),
        )
        
        for k in t.keys():
            if k == 't_idx' or k == 't_is_unique':
                t[k] = np.array(t[k], dtype='int32')
            else:
                t[k] = np.array(t[k], dtype='float32')
            t[k] = SimpleArray(t[k],  rs,name=k)
        
        return [farr, 
                t['t_idx'], t['t_energy'], t['t_pos'], t['t_time'], 
                t['t_pid'], t['t_spectator'], t['t_fully_contained'],
                t['t_rec_energy'], t['t_is_unique'] ], [], []

    def writeOutPrediction(self, predicted, features, truth, weights, outfilename, inputfile):
        outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'
        # print("hello", outfilename, inputfile)

        outdict = dict()
        outdict['predicted'] = predicted
        outdict['features'] = features
        outdict['truth'] = truth

        print("Writing to ", outfilename)
        with gzip.open(outfilename, "wb") as mypicklefile:
            pickle.dump(outdict, mypicklefile)
        print("Done")

    def writeOutPredictionDict(self, dumping_data, outfilename):
        '''
        this function should not be necessary... why break with DJC standards?
        '''
        if not str(outfilename).endswith('.bin.gz'):
            outfilename = os.path.splitext(outfilename)[0] + '.bin.gz'

        with gzip.open(outfilename, 'wb') as f2:
            pickle.dump(dumping_data, f2)

    def readPredicted(self, predfile):
        with gzip.open(predfile) as mypicklefile:
            return pickle.load(mypicklefile)
        
    
    
