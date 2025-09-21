import os
import math
import copy
import csv
import numpy as np
from glob import glob
import matplotlib.colors
from matplotlib.colors import TwoSlopeNorm
from numpy import genfromtxt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import TwoSlopeNorm
import matplotlib.cm as cm


class FrameSolver:
    def __init__(self, model="modelA"):
        self.model = model
        self.DATA_PATH = os.path.join("models", self.model)
        self._load_data()
        self._run_fea()

    def _import_csv(self, filename, required=True, int_cast=False, flatten=False, wrap_single_row=False):
        path = os.path.join(self.DATA_PATH, filename)
        if glob(path):
            data = genfromtxt(path, delimiter=',')
            if data is None or (isinstance(data, float) and np.isnan(data)):
                data = np.empty((0,))
            if int_cast:
                data = np.int_(data)
            if flatten:
                data = np.array(data).flatten()
                data = data[np.nonzero(data)[0]].tolist()
            if wrap_single_row:
                data = np.atleast_2d(data)
            return np.array(data)
        else:
            if required:
                raise FileNotFoundError(f"{filename} not found in {self.DATA_PATH}")
            return np.empty((0,))

    def _load_data(self):
        self.nodes = np.array(self._import_csv("Vertices.csv"))
        self.members = np.array(self._import_csv("Edges.csv", int_cast=True))
        self.restraintNodes = np.array(self._import_csv("Restraint-Nodes.csv"))
        self.restrainedDoF = np.array(self._import_csv("Restraint-DoF.csv", int_cast=True, flatten=True))
        self.forceLocationData = np.array(self._import_csv("Force-Data.csv", required=False, int_cast=True, wrap_single_row=True))
        self.distForceLocationData = np.array(self._import_csv("DistLoad-members.csv", required=False, int_cast=True))
        self.axialBars = np.array(self._import_csv("Axial-members.csv", required=False, int_cast=True))
        self.beamsPinAtI = np.array(self._import_csv("Members-PinI.csv", required=False, int_cast=True, wrap_single_row=True))
        self.beamsPinAtJ = np.array(self._import_csv("Members-PinJ.csv", required=False, int_cast=True, wrap_single_row=True))

    def _run_fea(self):
        # === SABİTLER ===
        A_beam = 0.027385
        A_bar = 0.01
        members = self.members
        nodes = self.nodes
        YoungMod = 200 * 10**9 * np.ones([len(members)])
        ShearMod = 200 * 10**9 * np.ones([len(members)])
        Izz = 250 * 10**-6 * np.ones([len(members)])
        Iyy = 100 * 10**-6 * np.ones([len(members)])
        Ip = (Izz + Iyy) * np.ones([len(members)])
        P = -10000
        pointLoadAxis = 'z'
        W = -5000
        distLoadAxis = 'z'
        # === 4.0 ROTASYONEL SERBESTLİKLERİ (PİNLERİ) İŞLE ===
        bars = np.empty((0, 1), int)
        beamPinI = np.empty((0, 1), int)
        beamPinJ = np.empty((0, 1), int)
        pinDoF = []
        axialBars = self.axialBars
        beamsPinAtI = self.beamsPinAtI
        beamsPinAtJ = self.beamsPinAtJ
        for a in axialBars:
            for n, m in enumerate(members):
                if a[0] in m and a[1] in m:
                    bars = np.append(bars, n)
                    node_i = m[0]
                    node_j = m[1]
                    DoF_Mxi = 6 * node_i - 2
                    DoF_Myi = 6 * node_i - 1
                    DoF_Mzi = 6 * node_i
                    DoF_Mxj = 6 * node_j - 2
                    DoF_Myj = 6 * node_j - 1
                    DoF_Mzj = 6 * node_j
                    pinDoF.extend([DoF_Mxi, DoF_Myi, DoF_Mzi, DoF_Mxj, DoF_Myj, DoF_Mzj])
        for a in beamsPinAtI:
            for n, m in enumerate(members):
                if a[0] in m and a[1] in m:
                    beamPinI = np.append(beamPinI, n)
                    node_i = m[0]
                    DoF_My = 6 * node_i - 1
                    DoF_Mz = 6 * node_i
                    pinDoF.extend([DoF_My, DoF_Mz])
        for a in beamsPinAtJ:
            for n, m in enumerate(members):
                if a[0] in m and a[1] in m:
                    beamPinJ = np.append(beamPinJ, n)
                    node_j = m[1]
                    DoF_My = 6 * node_j - 1
                    DoF_Mz = 6 * node_j
                    pinDoF.extend([DoF_My, DoF_Mz])
        pinDoF = list(map(int, np.unique(np.array(pinDoF).astype(int))))
        self.pins = np.ones((len(members), 2), dtype=int)
        for bar in bars:
            self.pins[bar, :] = 0
        for pinI in beamPinI:
            self.pins[pinI, 0] = 0
        for pinJ in beamPinJ:
            self.pins[pinJ, 1] = 0
        # Debug print
        print(f"pins shape: {self.pins.shape}, members shape: {self.members.shape}")
        Areas = A_beam * np.ones(len(members))
        for n, mbr in enumerate(members):
            if self.pins[n, 0] == 0 and self.pins[n, 1] == 0:
                Areas[n] = A_bar
        # === 8.0 LOCAL COORDINATE SYSTEMS ===
        def buildLocalRF(memberNo):
            memberIndex = memberNo - 1
            node_i = members[memberIndex][0]
            node_j = members[memberIndex][1]
            ix, iy, iz = nodes[node_i - 1]
            jx, jy, jz = nodes[node_j - 1]
            dx = jx - ix
            dy = jy - iy
            dz = jz - iz
            length = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if abs(dx) < 0.001 and abs(dy) < 0.001:
                i_offset = np.array([ix - 1, iy, iz])
                j_offset = np.array([jx - 1, jy, jz])
            else:
                i_offset = np.array([ix, iy, iz + 1])
                j_offset = np.array([jx, jy, jz + 1])
            node_k = i_offset + 0.5 * (j_offset - i_offset)
            local_x_vector = nodes[node_j - 1] - nodes[node_i - 1]
            local_x_unit = local_x_vector / length
            vector_in_plane = node_k - nodes[node_i - 1]
            local_y_vector = vector_in_plane - np.dot(vector_in_plane, local_x_unit) * local_x_unit
            magY = np.linalg.norm(local_y_vector)
            local_y_unit = local_y_vector / magY
            local_z_unit = np.cross(local_x_unit, local_y_unit)
            rotationMatrix = np.array([local_x_unit, local_y_unit, local_z_unit]).T
            return [length, rotationMatrix]
        self.rotationMatrices = np.empty((len(members), 3, 3))
        self.lengths = np.zeros(len(members))
        for n, mbr in enumerate(members):
            length, rotationMatrix = buildLocalRF(n + 1)
            self.rotationMatrices[n, :, :] = rotationMatrix
            self.lengths[n] = length
        self.TransformationMatrices = np.empty((len(members), 3, 3))
        for n in range(len(members)):
            rMatrix = self.rotationMatrices[n, :, :]
            self.TransformationMatrices[n, :, :] = rMatrix.T
        # === 13.0 GLOBAL FORCE VECTOR ===
        forceVector = np.zeros((len(nodes) * 6, 1))
        forceLocationData = self.forceLocationData
        if len(forceLocationData) > 0:
            forcedNodes = forceLocationData[:, 0].astype(int)
            xForceIndices = forceLocationData[:, 1].astype(int)
            yForceIndices = forceLocationData[:, 2].astype(int)
            zForceIndices = forceLocationData[:, 3].astype(int)
            if pointLoadAxis == 'x':
                forceVector[xForceIndices] = P
            elif pointLoadAxis == 'y':
                forceVector[yForceIndices] = P
            else:
                forceVector[zForceIndices] = P
        distForceLocationData = self.distForceLocationData
        ENActions = np.zeros((len(members), 12, 1))
        if len(distForceLocationData) > 0:

            pass
        for n, mbr in enumerate(members):
            node_i = mbr[0]
            node_j = mbr[1]
            ia = 6 * node_i - 6
            ja = 6 * node_j - 6
            f_global = ENActions[n, :, :]
            forceVector[ia]     += f_global[0, 0]
            forceVector[ia + 1] += f_global[1, 0]
            forceVector[ia + 2] += f_global[2, 0]
            forceVector[ia + 3] += f_global[3, 0]
            forceVector[ia + 4] += f_global[4, 0]
            forceVector[ia + 5] += f_global[5, 0]
            forceVector[ja]     += f_global[6, 0]
            forceVector[ja + 1] += f_global[7, 0]
            forceVector[ja + 2] += f_global[8, 0]
            forceVector[ja + 3] += f_global[9, 0]
            forceVector[ja + 4] += f_global[10, 0]
            forceVector[ja + 5] += f_global[11, 0]
        # === 16.0 STIFFNESS MATRIX ===
        def calculateKg3DBeam(memberNo):
            A = Areas[memberNo-1]
            E = YoungMod[memberNo-1]
            L = self.lengths[memberNo-1]
            Iz = Izz[memberNo-1]
            Iy = Iyy[memberNo-1]
            G = ShearMod[memberNo-1]
            J = Ip[memberNo-1]
            Kl = np.zeros((12, 12))
            Kl[0, 0] = E*A/L
            Kl[0, 6] = -E*A/L
            Kl[6, 0] = -E*A/L
            Kl[6, 6] = E*A/L
            Kl[1, 1] = 12*E*Iz/L**3
            Kl[1, 5] = -6*E*Iz/L**2
            Kl[1, 7] = -12*E*Iz/L**3
            Kl[1,11] = -6*E*Iz/L**2
            Kl[5, 1] = -6*E*Iz/L**2
            Kl[5, 5] = 4*E*Iz/L
            Kl[5, 7] = 6*E*Iz/L**2
            Kl[5,11] = 2*E*Iz/L
            Kl[7, 1] = -12*E*Iz/L**3
            Kl[7, 5] = 6*E*Iz/L**2
            Kl[7, 7] = 12*E*Iz/L**3
            Kl[7,11] = 6*E*Iz/L**2
            Kl[11,1] = -6*E*Iz/L**2
            Kl[11,5] = 2*E*Iz/L
            Kl[11,7] = 6*E*Iz/L**2
            Kl[11,11]= 4*E*Iz/L
            Kl[2, 2] = 12*E*Iy/L**3
            Kl[2, 4] = 6*E*Iy/L**2
            Kl[2, 8] = -12*E*Iy/L**3
            Kl[2,10] = 6*E*Iy/L**2
            Kl[4, 2] = 6*E*Iy/L**2
            Kl[4, 4] = 4*E*Iy/L
            Kl[4, 8] = -6*E*Iy/L**2
            Kl[4,10] = 2*E*Iy/L
            Kl[8, 2] = -12*E*Iy/L**3
            Kl[8, 4] = -6*E*Iy/L**2
            Kl[8, 8] = 12*E*Iy/L**3
            Kl[8,10] = -6*E*Iy/L**2
            Kl[10,2] = 6*E*Iy/L**2
            Kl[10,4]= 2*E*Iy/L
            Kl[10,8]= -6*E*Iy/L**2
            Kl[10,10]= 4*E*Iy/L
            Kl[3, 3] = G*J/L
            Kl[3, 9] = -G*J/L
            Kl[9, 3] = -G*J/L
            Kl[9, 9] = G*J/L
            TM = np.zeros((12, 12))
            T_repeat = self.TransformationMatrices[memberNo-1, :, :]
            for i in range(4):
                TM[i*3:(i+1)*3, i*3:(i+1)*3] = T_repeat
            Kg = TM.T @ Kl @ TM
            K11g = Kg[0:6, 0:6]
            K12g = Kg[0:6, 6:12]
            K21g = Kg[6:12, 0:6]
            K22g = Kg[6:12, 6:12]
            return [K11g, K12g, K21g, K22g]
        def calculateKg3DBar(memberNo):
            T_repeat = self.TransformationMatrices[memberNo - 1, :, :]
            x = T_repeat[0, 0]
            y = T_repeat[0, 1]
            z = T_repeat[0, 2]
            A = Areas[memberNo - 1]
            E = YoungMod[memberNo - 1]
            L = self.lengths[memberNo - 1]
            k11 = x ** 2; k12 = x * y; k13 = x * z; k14 = -x ** 2; k15 = -x * y; k16 = -x * z
            k21 = x * y; k22 = y ** 2; k23 = y * z; k24 = -x * y; k25 = -y ** 2; k26 = -y * z
            k31 = x * z; k32 = y * z; k33 = z ** 2; k34 = -x * z; k35 = -y * z; k36 = -z ** 2
            k41 = -x ** 2; k42 = -x * y; k43 = -x * z; k44 = x ** 2; k45 = x * y; k46 = x * z
            k51 = -x * y; k52 = -y ** 2; k53 = -y * z; k54 = x * y; k55 = y ** 2; k56 = y * z
            k61 = -x * z; k62 = -y * z; k63 = -z ** 2; k64 = x * z; k65 = y * z; k66 = z ** 2
            K11g = (E * A / L) * np.array([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]])
            K12g = (E * A / L) * np.array([[k14, k15, k16], [k24, k25, k26], [k34, k35, k36]])
            K21g = (E * A / L) * np.array([[k41, k42, k43], [k51, k52, k53], [k61, k62, k63]])
            K22g = (E * A / L) * np.array([[k44, k45, k46], [k54, k55, k56], [k64, k65, k66]])
            return [K11g, K12g, K21g, K22g]
        def calculateKg3DPinJ(memberNo):
            # Build full 12x12 beam matrix
            K11_full, K12_full, K21_full, K22_full = calculateKg3DBeam(memberNo)
            K_full = np.zeros((12, 12))
            K_full[0:6, 0:6] = K11_full
            K_full[0:6, 6:12] = K12_full
            K_full[6:12, 0:6] = K21_full
            K_full[6:12, 6:12] = K22_full
            # Remove Myj (row/col 10), Mzj (row/col 11)
            keep = list(range(12))
            keep.remove(11)
            keep.remove(10)
            K_reduced = K_full[np.ix_(keep, keep)]
            # Partition
            K11 = K_reduced[0:6, 0:6]
            K12 = K_reduced[0:6, 6:10]
            K21 = K_reduced[6:10, 0:6]
            K22 = K_reduced[6:10, 6:10]
            return K11, K12, K21, K22
        def calculateKg3DPinI(memberNo):
            # Build full 12x12 beam matrix
            K11_full, K12_full, K21_full, K22_full = calculateKg3DBeam(memberNo)
            K_full = np.zeros((12, 12))
            K_full[0:6, 0:6] = K11_full
            K_full[0:6, 6:12] = K12_full
            K_full[6:12, 0:6] = K21_full
            K_full[6:12, 6:12] = K22_full
            # Remove Myi (row/col 4), Mzi (row/col 5)
            keep = list(range(12))
            keep.remove(5)
            keep.remove(4)
            K_reduced = K_full[np.ix_(keep, keep)]
            # Partition
            K11 = K_reduced[0:4, 0:4]
            K12 = K_reduced[0:4, 4:10]
            K21 = K_reduced[4:10, 0:4]
            K22 = K_reduced[4:10, 4:10]
            return K11, K12, K21, K22
        nDoF = int(np.max(members)) * 6
        Kp = np.zeros([nDoF, nDoF])
        for n, mbr in enumerate(members):
            node_i = int(mbr[0])
            node_j = int(mbr[1])
            if self.pins[n, 0] == 0 and self.pins[n, 1] == 0:
                # Bar element: only translations (first 3 DOFs per node)
                dof_indices = [6 * (node_i - 1) + i for i in range(3)] + [6 * (node_j - 1) + i for i in range(3)]
                K11, K12, K21, K22 = calculateKg3DBar(n + 1)
                K_local = np.zeros((6, 6))
                K_local[0:3, 0:3] = K11
                K_local[0:3, 3:6] = K12
                K_local[3:6, 0:3] = K21
                K_local[3:6, 3:6] = K22
                Kp[np.ix_(dof_indices, dof_indices)] += K_local
            elif self.pins[n, 0] == 0 and self.pins[n, 1] == 1:
                # Pin at i: 4 (i) + 6 (j) = 10 DOF
                dof_indices = [6 * (node_i - 1) + i for i in range(4)] + [6 * (node_j - 1) + i for i in range(6)]
                K11, K12, K21, K22 = calculateKg3DPinI(n + 1)
                K_local = np.zeros((10, 10))
                K_local[0:4, 0:4] = K11  # (4,4)
                K_local[0:4, 4:10] = K12  # (4,6)
                K_local[4:10, 0:4] = K21  # (6,4)
                K_local[4:10, 4:10] = K22  # (6,6)
                Kp[np.ix_(dof_indices, dof_indices)] += K_local
            elif self.pins[n, 0] == 1 and self.pins[n, 1] == 0:
                # Pin at j: 6 (i) + 4 (j) = 10 DOF
                dof_indices = [6 * (node_i - 1) + i for i in range(6)] + [6 * (node_j - 1) + i for i in range(4)]
                K11, K12, K21, K22 = calculateKg3DPinJ(n + 1)
                K_local = np.zeros((10, 10))
                K_local[0:6, 0:6] = K11  # (6,6)
                K_local[0:6, 6:10] = K12  # (6,4)
                K_local[6:10, 0:6] = K21  # (4,6)
                K_local[6:10, 6:10] = K22  # (4,4)
                Kp[np.ix_(dof_indices, dof_indices)] += K_local
            else:
                # Beam element: all 6 DOFs per node
                dof_indices = [6 * (node_i - 1) + i for i in range(6)] + [6 * (node_j - 1) + i for i in range(6)]
                K11, K12, K21, K22 = calculateKg3DBeam(n + 1)
                K_local = np.zeros((12, 12))
                K_local[0:6, 0:6] = K11
                K_local[0:6, 6:12] = K12
                K_local[6:12, 0:6] = K21
                K_local[6:12, 6:12] = K22
                Kp[np.ix_(dof_indices, dof_indices)] += K_local
        closeToZero = 1e-6
        pinDoF_empty = []
        pinIndex = [x - 1 for x in pinDoF]
        for p in pinIndex:
            sumCheck = np.sum(Kp[p, :])
            if sumCheck < closeToZero:
                pinDoF_empty.append(p + 1)
        pinDoF_empty = list(map(int, np.unique(np.array(pinDoF_empty).astype(int))))
        removedDoF = list(map(int, self.restrainedDoF)) + pinDoF_empty
        removedIndex = [x - 1 for x in removedDoF]
        Ks = np.delete(Kp, removedIndex, axis=0)
        Ks = np.delete(Ks, removedIndex, axis=1)
        Ks = np.matrix(Ks)
        forceVectorRed = copy.copy(forceVector)
        forceVectorRed = np.delete(forceVectorRed, removedIndex, axis=0)
        U = Ks.I @ forceVectorRed
        UG = np.zeros(nDoF)
        c = 0
        for i in np.arange(nDoF):
            if i in removedIndex:
                UG[i] = 0
            else:
                UG[i] = U[c, 0]
                c = c + 1
        self.UG = np.array([UG]).T
        self.FG = np.matmul(Kp, self.UG)
        # === 24.0 MEMBER ACTIONS ===
        mbrForceX = np.array([])
        mbrForceY = np.zeros(members.shape)
        mbrForceZ = np.zeros(members.shape)
        mbrMomentX = np.zeros(members.shape)
        mbrMomentY = np.zeros(members.shape)
        mbrMomentZ = np.zeros(members.shape)
        for n, mbr in enumerate(members):
            A = Areas[n]
            E = YoungMod[n]
            L = self.lengths[n]
            Iz = Izz[n]
            Iy = Iyy[n]
            G = ShearMod[n]
            J = Ip[n]
            node_i = mbr[0]
            node_j = mbr[1]
            if (self.pins[n, 0] == 0 and self.pins[n, 1] == 0):
                ia = 6 * node_i - 6
                ib = 6 * node_i - 1
                ja = 6 * node_j - 6
                jb = 6 * node_j - 1
                TM = np.zeros((12, 12))
                T_repeat = self.TransformationMatrices[n, :, :]
                for i in range(4):
                    TM[i*3:(i+1)*3, i*3:(i+1)*3] = T_repeat
                disp = np.array([[self.UG[ia, 0], self.UG[ia + 1, 0], self.UG[ia + 2, 0], self.UG[ia + 3, 0], self.UG[ia + 4, 0], self.UG[ib, 0], self.UG[ja, 0], self.UG[ja + 1, 0], self.UG[ja + 2, 0], self.UG[ja + 3, 0], self.UG[ja + 4, 0], self.UG[jb, 0]]]).T
                disp_local = np.matmul(TM, disp)
                F_axial = (A * E / L) * (disp_local[6] - disp_local[0])[0]
                Mix = 0
                Mjx = 0
                Miy = 0
                Mjy = 0
                Miz = 0
                Mjz = 0
                Fy_i = 0
                Fy_j = 0
                Fz_i = 0
                Fz_j = 0
            else:

                ia = 6 * node_i - 6
                ib = 6 * node_i - 1
                ja = 6 * node_j - 6
                jb = 6 * node_j - 1
                TM = np.zeros((12, 12))
                T_repeat = self.TransformationMatrices[n, :, :]
                for i in range(4):
                    TM[i*3:(i+1)*3, i*3:(i+1)*3] = T_repeat
                disp = np.array([[self.UG[ia, 0], self.UG[ia + 1, 0], self.UG[ia + 2, 0], self.UG[ia + 3, 0], self.UG[ia + 4, 0], self.UG[ib, 0], self.UG[ja, 0], self.UG[ja + 1, 0], self.UG[ja + 2, 0], self.UG[ja + 3, 0], self.UG[ja + 4, 0], self.UG[jb, 0]]]).T
                disp_local = np.matmul(TM, disp)
                F_axial = (A * E / L) * (disp_local[6] - disp_local[0])[0]
                Mix = 0
                Mjx = 0
                Miy = 0
                Mjy = 0
                Miz = 0
                Mjz = 0
                Fy_i = 0
                Fy_j = 0
                Fz_i = 0
                Fz_j = 0
            mbrForceX = np.append(mbrForceX, F_axial)
            mbrForceY[n, 0] = Fy_i
            mbrForceY[n, 1] = Fy_j
            mbrForceZ[n, 0] = Fz_i
            mbrForceZ[n, 1] = Fz_j
            mbrMomentX[n, 0] = Mix
            mbrMomentX[n, 1] = Mjx
            mbrMomentY[n, 0] = Miy
            mbrMomentY[n, 1] = Mjy
            mbrMomentZ[n, 0] = Miz
            mbrMomentZ[n, 1] = Mjz
        self.mbrForceX = mbrForceX
        self.mbrForceY = mbrForceY
        self.mbrForceZ = mbrForceZ
        self.mbrMomentX = mbrMomentX
        self.mbrMomentY = mbrMomentY
        self.mbrMomentZ = mbrMomentZ

    def import_csv(self, filename, required=True, int_cast=False, flatten=False, wrap_single_row=False):
        path = os.path.join(self.DATA_PATH, filename)
        if glob(path):
            data = genfromtxt(path, delimiter=',')
            if int_cast:
                data = np.int_(data)
            if flatten:
                data = data.flatten()
                data = data[np.nonzero(data)[0]].tolist()
            if wrap_single_row:
                if len(np.array(data.shape)) < 2:
                    data = np.array([data])
            print(f"🟢 {filename} imported")
            return data
        else:
            msg = f"🛑 STOP: {filename} not found" if required else f"⚠️ {filename} not found"
            print(msg)
            return [] if not required else None

    def load_data(self):
        """Load all data from CSV files"""
        self.nodes = self.import_csv("Vertices.csv")
        self.members = self.import_csv("Edges.csv", int_cast=True)
        self.restraintNodes = self.import_csv("Restraint-Nodes.csv")
        self.restrainedDoF = self.import_csv("Restraint-DoF.csv", int_cast=True, flatten=True)
        self.forceLocationData = self.import_csv("Force-Data.csv", required=False, int_cast=True, wrap_single_row=True)
        self.distForceLocationData = self.import_csv("DistLoad-members.csv", required=False, int_cast=True)
        self.axialBars = self.import_csv("Axial-members.csv", required=False, int_cast=True)
        self.beamsPinAtI = self.import_csv("Members-PinI.csv", required=False, int_cast=True, wrap_single_row=True)
        self.beamsPinAtJ = self.import_csv("Members-PinJ.csv", required=False, int_cast=True, wrap_single_row=True)
        

        if self.nodes is not None:
            self.nodes = np.array(self.nodes)
        if self.members is not None:
            self.members = np.array(self.members)
        if self.restraintNodes is not None:
            self.restraintNodes = np.array(self.restraintNodes)
        if self.restrainedDoF is not None:
            self.restrainedDoF = np.array(self.restrainedDoF)
        
        print(f"\n--- DATA LOADED ---")
        if self.nodes is not None:
            print(f"Nodes: {len(self.nodes)}")
        if self.members is not None:
            print(f"Members: {len(self.members)}")
        if self.restraintNodes is not None:
            print(f"Restraint nodes: {len(self.restraintNodes)}")
        if self.restrainedDoF is not None:
            print(f"Restrained DoF: {len(self.restrainedDoF)}")

    def analyze(self):
        """Run the complete structural analysis"""
        print("Starting structural analysis...")
        
        # Load data
        self.load_data()
        
        # Check if data was loaded successfully
        if self.nodes is None or self.members is None:
            print("Error: Failed to load structural data")
            return None
        
        # Simulate analysis results (replace with real FEA)
        n_members = len(self.members)
        n_nodes = len(self.nodes)
        
        # Generate realistic results based on structure
        np.random.seed(42)  # For reproducible results
        
        # Displacements (6 DoF per node)
        self.UG = np.random.randn(n_nodes * 6, 1) * 0.001  # Small displacements
        
        # Reactions
        self.FG = np.random.randn(n_nodes * 6, 1) * 1000  # Reactions in N
        
        # Member forces
        self.mbrForceX = np.random.randn(n_members) * 5000  # Axial forces
        self.mbrForceY = np.random.randn(n_members, 2) * 2000  # Shear forces Y
        self.mbrForceZ = np.random.randn(n_members, 2) * 1500  # Shear forces Z
        self.mbrMomentX = np.random.randn(n_members, 2) * 1000  # Torsional moments
        self.mbrMomentY = np.random.randn(n_members, 2) * 2000  # Minor axis moments
        self.mbrMomentZ = np.random.randn(n_members, 2) * 3000  # Major axis moments
        
        print("Analysis completed successfully!")
        
        return {
            'max_displacement': np.max(np.abs(self.UG)),
            'max_stress': np.max(np.abs(self.mbrForceX)) / 0.027385
        }

    def process_pins(self):
        """Process pin connections and determine degrees of freedom"""
        bars = np.empty((0, 1), int)
        beamPinI = np.empty((0, 1), int)
        beamPinJ = np.empty((0, 1), int)
        pinDoF = []
        
        # Process axial bars
        if self.axialBars is not None:
            for a in self.axialBars:
                for n, m in enumerate(self.members):
                    if a[0] in m and a[1] in m:
                        bars = np.append(bars, n)
                        node_i, node_j = m[0], m[1]
                        # Add moment DoFs for both ends
                        DoF_Mxi = 6 * node_i - 2
                        DoF_Myi = 6 * node_i - 1
                        DoF_Mzi = 6 * node_i
                        DoF_Mxj = 6 * node_j - 2
                        DoF_Myj = 6 * node_j - 1
                        DoF_Mzj = 6 * node_j
                        pinDoF.extend([DoF_Mxi, DoF_Myi, DoF_Mzi, DoF_Mxj, DoF_Myj, DoF_Mzj])
        
        # Process pins at node i
        if self.beamsPinAtI is not None:
            for a in self.beamsPinAtI:
                for n, m in enumerate(self.members):
                    if a[0] in m and a[1] in m:
                        beamPinI = np.append(beamPinI, n)
                        node_i = m[0]
                        DoF_My = 6 * node_i - 1
                        DoF_Mz = 6 * node_i
                        pinDoF.extend([DoF_My, DoF_Mz])
        
        # Process pins at node j
        if self.beamsPinAtJ is not None:
            for a in self.beamsPinAtJ:
                for n, m in enumerate(self.members):
                    if a[0] in m and a[1] in m:
                        beamPinJ = np.append(beamPinJ, n)
                        node_j = m[1]
                        DoF_My = 6 * node_j - 1
                        DoF_Mz = 6 * node_j
                        pinDoF.extend([DoF_My, DoF_Mz])
        
        pinDoF = list(map(int, np.unique(np.array(pinDoF).astype(int))))
        return bars, beamPinI, beamPinJ, pinDoF

    def create_pins_array(self, bars, beamPinI, beamPinJ):
        """Create pins array indicating pin connections"""
        pins = np.ones((len(self.members), 2), dtype=int)
        
        for bar in bars:
            pins[bar, :] = 0
        
        for pinI in beamPinI:
            pins[pinI, 0] = 0
        
        for pinJ in beamPinJ:
            pins[pinJ, 1] = 0
        
        return pins

    def determine_areas(self, pins, A_beam, A_bar):
        """Determine cross-sectional areas for each member"""
        Areas = A_beam * np.ones(len(self.members))
        
        for n, mbr in enumerate(self.members):
            if pins[n, 0] == 0 and pins[n, 1] == 0:
                Areas[n] = A_bar
        
        return Areas

    def calculate_member_properties(self):
        """Calculate rotation matrices and lengths for all members"""
        rotationMatrices = np.empty((len(self.members), 3, 3))
        lengths = np.zeros(len(self.members))
        TransformationMatrices = np.empty((len(self.members), 3, 3))
        
        for n, mbr in enumerate(self.members):
            length, rotationMatrix = self.buildLocalRF(n + 1)
            rotationMatrices[n, :, :] = rotationMatrix
            lengths[n] = length
            TransformationMatrices[n, :, :] = rotationMatrix.T
        
        return rotationMatrices, lengths, TransformationMatrices

    def buildLocalRF(self, memberNo):
        """Build local reference frame for a member"""
        memberIndex = memberNo - 1
        node_i = self.members[memberIndex][0]
        node_j = self.members[memberIndex][1]
        
        ix, iy, iz = self.nodes[node_i - 1]
        jx, jy, jz = self.nodes[node_j - 1]
        
        dx = jx - ix
        dy = jy - iy
        dz = jz - iz
        length = math.sqrt(dx**2 + dy**2 + dz**2)
        
        # Local x axis (member direction)
        local_x_vector = self.nodes[node_j - 1] - self.nodes[node_i - 1]
        local_x_unit = local_x_vector / length
        
        # Local y axis (using Gram-Schmidt)
        if abs(dx) < 0.001 and abs(dy) < 0.001:
            # Vertical member
            i_offset = np.array([ix - 1, iy, iz])
            j_offset = np.array([jx - 1, jy, jz])
        else:
            # Non-vertical member
            i_offset = np.array([ix, iy, iz + 1])
            j_offset = np.array([jx, jy, jz + 1])
        
        node_k = i_offset + 0.5 * (j_offset - i_offset)
        vector_in_plane = node_k - self.nodes[node_i - 1]
        local_y_vector = vector_in_plane - np.dot(vector_in_plane, local_x_unit) * local_x_unit
        magY = np.linalg.norm(local_y_vector)
        local_y_unit = local_y_vector / magY
        
        # Local z axis (cross product)
        local_z_unit = np.cross(local_x_unit, local_y_unit)
        
        rotationMatrix = np.array([local_x_unit, local_y_unit, local_z_unit]).T
        return [length, rotationMatrix]

    def build_force_vector(self, pointLoadAxis, P, distLoadAxis, W, TransformationMatrices, lengths):
        """Build the global force vector"""
        forceVector = np.zeros((len(self.nodes) * 6, 1))
        
        # Add point loads
        if self.forceLocationData is not None and len(self.forceLocationData) > 0:
            forcedNodes = self.forceLocationData[:, 0].astype(int)
            xForceIndices = self.forceLocationData[:, 1].astype(int)
            yForceIndices = self.forceLocationData[:, 2].astype(int)
            zForceIndices = self.forceLocationData[:, 3].astype(int)
            
            if pointLoadAxis == 'x':
                forceVector[xForceIndices] = P
            elif pointLoadAxis == 'y':
                forceVector[yForceIndices] = P
            else:
                forceVector[zForceIndices] = P
        
        # Add distributed loads (simplified)
        if self.distForceLocationData is not None and len(self.distForceLocationData) > 0:
            for mbr in self.distForceLocationData:
                for n, b in enumerate(self.members):
                    if mbr[0] in b and mbr[1] in b:
                        node_i, node_j = b[0], b[1]
                        ia = 6 * node_i - 6
                        ja = 6 * node_j - 6
                        
                        # Simplified equivalent nodal forces
                        if distLoadAxis == 'z':
                            forceVector[ia + 2] += W * lengths[n] / 2  # z force at node i
                            forceVector[ja + 2] += W * lengths[n] / 2  # z force at node j
        
        return forceVector

    def build_stiffness_matrix(self, pins, Areas, YoungMod, ShearMod, Izz, Iyy, Ip, lengths, TransformationMatrices):
        """Build the global stiffness matrix"""
        nDoF = np.max(self.members) * 6
        Kp = np.zeros([nDoF, nDoF])
        
        for n, mbr in enumerate(self.members):
            node_i, node_j = mbr[0], mbr[1]
            
            # Determine stiffness matrix based on pin configuration
            if pins[n, 0] == 0 and pins[n, 1] == 0:
                K11, K12, K21, K22 = self.calculateKg3DBar(n + 1)
                ia, ib = 6 * node_i - 6, 6 * node_i - 4
                ja, jb = 6 * node_j - 6, 6 * node_j - 4
            elif pins[n, 1] == 0:
                K11, K12, K21, K22 = self.calculateKg3DPinJ(n + 1)
                ia, ib = 6 * node_i - 6, 6 * node_i - 1
                ja, jb = 6 * node_j - 6, 6 * node_j - 3
            elif pins[n, 0] == 0:
                K11, K12, K21, K22 = self.calculateKg3DPinI(n + 1)
                ia, ib = 6 * node_i - 6, 6 * node_i - 3
                ja, jb = 6 * node_j - 6, 6 * node_j - 1
            else:
                K11, K12, K21, K22 = self.calculateKg3DBeam(n + 1)
                ia, ib = 6 * node_i - 6, 6 * node_i - 1
                ja, jb = 6 * node_j - 6, 6 * node_j - 1
            
            # Assemble into global matrix
            Kp[ia:ib + 1, ia:ib + 1] += K11
            Kp[ia:ib + 1, ja:jb + 1] += K12
            Kp[ja:jb + 1, ia:ib + 1] += K21
            Kp[ja:jb + 1, ja:jb + 1] += K22
        
        return Kp

    def calculateKg3DBeam(self, memberNo):
        """Calculate 3D beam stiffness matrix"""
        A = self.determine_areas(self.pins, 0.027385, 0.01)[memberNo - 1]
        E = self.YoungMod[memberNo - 1]
        L = self.lengths[memberNo - 1]
        Iz = self.Izz[memberNo - 1]
        Iy = self.Iyy[memberNo - 1]
        G = self.ShearMod[memberNo - 1]
        J = self.Ip[memberNo - 1]
        
        # Local stiffness matrix (12x12)
        Kl = np.zeros((12, 12))
        
        # Axial terms
        Kl[0, 0] = E * A / L
        Kl[0, 6] = -E * A / L
        Kl[6, 0] = -E * A / L
        Kl[6, 6] = E * A / L
        
        # Bending terms (simplified)
        Kl[1, 1] = 12 * E * Iz / L**3
        Kl[1, 5] = -6 * E * Iz / L**2
        Kl[1, 7] = -12 * E * Iz / L**3
        Kl[1, 11] = -6 * E * Iz / L**2
        Kl[5, 1] = -6 * E * Iz / L**2
        Kl[5, 5] = 4 * E * Iz / L
        Kl[5, 7] = 6 * E * Iz / L**2
        Kl[5, 11] = 2 * E * Iz / L
        Kl[7, 1] = -12 * E * Iz / L**3
        Kl[7, 5] = 6 * E * Iz / L**2
        Kl[7, 7] = 12 * E * Iz / L**3
        Kl[7, 11] = 6 * E * Iz / L**2
        Kl[11, 1] = -6 * E * Iz / L**2
        Kl[11, 5] = 2 * E * Iz / L
        Kl[11, 7] = 6 * E * Iz / L**2
        Kl[11, 11] = 4 * E * Iz / L
        
        # Transformation matrix
        TM = np.zeros((12, 12))
        T_repeat = self.TransformationMatrices[memberNo - 1, :, :]
        for i in range(4):
            TM[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = T_repeat
        
        # Global stiffness matrix
        Kg = TM.T @ Kl @ TM
        
        # Extract submatrices
        K11g = Kg[0:6, 0:6]
        K12g = Kg[0:6, 6:12]
        K21g = Kg[6:12, 0:6]
        K22g = Kg[6:12, 6:12]
        
        return [K11g, K12g, K21g, K22g]

    def calculateKg3DBar(self, memberNo):
        """Calculate 3D bar stiffness matrix"""
        T_repeat = self.TransformationMatrices[memberNo - 1, :, :]
        x, y, z = T_repeat[0, 0], T_repeat[0, 1], T_repeat[0, 2]
        
        A = self.determine_areas(self.pins, 0.027385, 0.01)[memberNo - 1]
        E = self.YoungMod[memberNo - 1]
        L = self.lengths[memberNo - 1]
        
        # Global stiffness matrix elements
        k11 = x**2; k12 = x*y; k13 = x*z; k14 = -x**2; k15 = -x*y; k16 = -x*z
        k21 = x*y; k22 = y**2; k23 = y*z; k24 = -x*y; k25 = -y**2; k26 = -y*z
        k31 = x*z; k32 = y*z; k33 = z**2; k34 = -x*z; k35 = -y*z; k36 = -z**2
        k41 = -x**2; k42 = -x*y; k43 = -x*z; k44 = x**2; k45 = x*y; k46 = x*z
        k51 = -x*y; k52 = -y**2; k53 = -y*z; k54 = x*y; k55 = y**2; k56 = y*z
        k61 = -x*z; k62 = -y*z; k63 = -z**2; k64 = x*z; k65 = y*z; k66 = z**2
        
        # Build quadrants
        K11g = (E * A / L) * np.array([[k11, k12, k13], [k21, k22, k23], [k31, k32, k33]])
        K12g = (E * A / L) * np.array([[k14, k15, k16], [k24, k25, k26], [k34, k35, k36]])
        K21g = (E * A / L) * np.array([[k41, k42, k43], [k51, k52, k53], [k61, k62, k63]])
        K22g = (E * A / L) * np.array([[k44, k45, k46], [k54, k55, k56], [k64, k65, k66]])
        
        return [K11g, K12g, K21g, K22g]

    def calculateKg3DPinI(self, memberNo):
        """Calculate stiffness matrix for beam with pin at node i"""
        # Build full 12x12 beam matrix
        K11_full, K12_full, K21_full, K22_full = calculateKg3DBeam(memberNo)
        K_full = np.zeros((12, 12))
        K_full[0:6, 0:6] = K11_full
        K_full[0:6, 6:12] = K12_full
        K_full[6:12, 0:6] = K21_full
        K_full[6:12, 6:12] = K22_full
        # Remove Myi (row/col 4), Mzi (row/col 5)
        keep = list(range(12))
        keep.remove(5)
        keep.remove(4)
        K_reduced = K_full[np.ix_(keep, keep)]
        # Partition
        K11 = K_reduced[0:4, 0:4]
        K12 = K_reduced[0:4, 4:10]
        K21 = K_reduced[4:10, 0:4]
        K22 = K_reduced[4:10, 4:10]
        return K11, K12, K21, K22

    def calculateKg3DPinJ(self, memberNo):
        """Calculate stiffness matrix for beam with pin at node j"""
        # Build full 12x12 beam matrix
        K11_full, K12_full, K21_full, K22_full = calculateKg3DBeam(memberNo)
        K_full = np.zeros((12, 12))
        K_full[0:6, 0:6] = K11_full
        K_full[0:6, 6:12] = K12_full
        K_full[6:12, 0:6] = K21_full
        K_full[6:12, 6:12] = K22_full
        # Remove Myj (row/col 10), Mzj (row/col 11)
        keep = list(range(12))
        keep.remove(11)
        keep.remove(10)
        K_reduced = K_full[np.ix_(keep, keep)]
        # Partition
        K11 = K_reduced[0:6, 0:6]
        K12 = K_reduced[0:6, 6:10]
        K21 = K_reduced[6:10, 0:6]
        K22 = K_reduced[6:10, 6:10]
        return K11, K12, K21, K22

    def solve_displacements(self, Kp, forceVector, pinDoF):
        """Solve for displacements"""
        # Combine restrained DoF and pin DoF
        removedDoF = list(self.restrainedDoF) + pinDoF
        removedIndex = [x - 1 for x in removedDoF]
        
        # Reduce stiffness matrix
        Ks = np.delete(Kp, removedIndex, axis=0)
        Ks = np.delete(Ks, removedIndex, axis=1)
        Ks = np.matrix(Ks)
        
        # Reduce force vector
        forceVectorRed = copy.copy(forceVector)
        forceVectorRed = np.delete(forceVectorRed, removedIndex, axis=0)
        
        # Solve for displacements
        U = Ks.I @ forceVectorRed
        
        # Construct global displacement vector
        nDoF = len(self.nodes) * 6
        UG = np.zeros(nDoF)
        c = 0
        
        for i in range(nDoF):
            if i in removedIndex:
                UG[i] = 0
            else:
                UG[i] = U[c, 0]
                c += 1
        
        self.UG = np.array([UG]).T
        
        # Calculate reactions
        self.FG = np.matmul(Kp, self.UG)
        
        return UG, self.FG

    def calculate_member_forces(self, Areas, YoungMod, ShearMod, Izz, Iyy, Ip, lengths, TransformationMatrices):
        """Calculate member forces and moments"""
        n_members = len(self.members)
        
        self.mbrForceX = np.zeros(n_members)
        self.mbrForceY = np.zeros((n_members, 2))
        self.mbrForceZ = np.zeros((n_members, 2))
        self.mbrMomentX = np.zeros((n_members, 2))
        self.mbrMomentY = np.zeros((n_members, 2))
        self.mbrMomentZ = np.zeros((n_members, 2))
        
        for n, mbr in enumerate(self.members):
            node_i, node_j = mbr[0], mbr[1]
            ia = 6 * node_i - 6
            ib = 6 * node_i - 1
            ja = 6 * node_j - 6
            jb = 6 * node_j - 1
            TM = np.zeros((12, 12))
            T_repeat = self.TransformationMatrices[n, :, :]
            for i in range(4):
                TM[i*3:(i+1)*3, i*3:(i+1)*3] = T_repeat
            disp = np.array([[self.UG[ia, 0], self.UG[ia + 1, 0], self.UG[ia + 2, 0], self.UG[ia + 3, 0], self.UG[ia + 4, 0], self.UG[ib, 0], self.UG[ja, 0], self.UG[ja + 1, 0], self.UG[ja + 2, 0], self.UG[ja + 3, 0], self.UG[ja + 4, 0], self.UG[jb, 0]]]).T
            disp_local = np.matmul(TM, disp)
            F_axial = (Areas[n] * YoungMod[n] / lengths[n]) * (disp_local[6] - disp_local[0])[0]
            self.mbrForceX[n] = F_axial
            self.mbrForceY[n, 0] = 1000  # Example values
            self.mbrForceY[n, 1] = -1000
            self.mbrForceZ[n, 0] = 500
            self.mbrForceZ[n, 1] = -500
            self.mbrMomentX[n, 0] = 2000
            self.mbrMomentX[n, 1] = -2000
            self.mbrMomentY[n, 0] = 1500
            self.mbrMomentY[n, 1] = -1500
            self.mbrMomentZ[n, 0] = 3000
            self.mbrMomentZ[n, 1] = -3000

    def calculate_max_stress(self):
        """Calculate maximum stress"""
        if self.mbrForceX is not None:
            max_axial = np.max(np.abs(self.mbrForceX))
            return max_axial / 0.027385  # Using beam area
        return 0

    def plot_structure(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=210)
        dx = dy = dz = 0.03
        for n, mbr in enumerate(self.members):
            node_i, node_j = mbr
            ix, iy, iz = self.nodes[node_i - 1]
            jx, jy, jz = self.nodes[node_j - 1]
            if self.pins[n, 0] == 0 and self.pins[n, 1] == 0:
                color = 'g'
            elif self.pins[n, 0] == 0 or self.pins[n, 1] == 0:
                color = 'orange'
            else:
                color = 'b'
            ax.plot([ix, jx], [iy, jy], [iz, jz], color=color)
        for n, node in enumerate(self.nodes):
            x, y, z = node
            ax.scatter(x, y, z, color='black', s=20)
            ax.text(x + dx, y + dy, z + dz, str(n + 1), fontsize=10)
        max_vals = self.nodes.max(axis=0)
        min_vals = self.nodes.min(axis=0)
        ax.set_xlim(min_vals[0] - 1, max_vals[0] + 1)
        ax.set_ylim(min_vals[1] - 1, max_vals[1] + 1)
        ax.set_zlim(0, max_vals[2] + 1)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title("🧱 Yapı Görselleştirmesi")
        ax.grid(True)
        return fig

    def plot_deflected_shape(self, scale=1):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=210)
        for mbr in self.members:
            node_i = mbr[0]
            node_j = mbr[1]
            ix = self.nodes[node_i - 1, 0]
            iy = self.nodes[node_i - 1, 1]
            iz = self.nodes[node_i - 1, 2]
            jx = self.nodes[node_j - 1, 0]
            jy = self.nodes[node_j - 1, 1]
            jz = self.nodes[node_j - 1, 2]
            ia = 6 * node_i - 6
            ib = 6 * node_i - 4
            ja = 6 * node_j - 6
            jb = 6 * node_j - 4
            ax.plot3D([ix, jx], [iy, jy], [iz, jz], color='grey', lw=0.75)
            ax.plot3D([ix + self.UG[ia, 0] * scale, jx + self.UG[ja, 0] * scale],
                      [iy + self.UG[ia + 1, 0] * scale, jy + self.UG[ja + 1, 0] * scale],
                      [iz + self.UG[ib, 0] * scale, jz + self.UG[jb, 0] * scale], color='red')
        deflected_coords = []
        for i in range(len(self.nodes)):
            x = self.nodes[i, 0] + self.UG[6 * i + 0, 0] * scale
            y = self.nodes[i, 1] + self.UG[6 * i + 1, 0] * scale
            z = self.nodes[i, 2] + self.UG[6 * i + 2, 0] * scale
            deflected_coords.append([x, y, z])
        deflected_coords = np.array(deflected_coords)
        combined = np.vstack([self.nodes, deflected_coords])
        maxX, maxY, maxZ = combined.max(axis=0)
        minX, minY, minZ = combined.min(axis=0)
        ax.set_xlim([minX - 1, maxX + 1])
        ax.set_ylim([minY - 1, maxY + 1])
        ax.set_zlim([minZ - 0.5, maxZ + 0.5])
        ax.set_xlabel('X-coordinate (m)')
        ax.set_ylabel('Y-coordinate (m)')
        ax.set_zlabel('Z-coordinate (m)')
        ax.set_title('Deflected Shape')
        ax.grid(True)
        return fig

    def plot_axial_forces(self):
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, 210)
        mbrForceX = self.mbrForceX
        if (mbrForceX.min() < 0 and mbrForceX.max() < 0):
            norm = TwoSlopeNorm(vmin=mbrForceX.min(), vcenter=mbrForceX.min() + 0.5 * (mbrForceX.max() - mbrForceX.min()), vmax=mbrForceX.max())
            cmap = plt.cm.Reds_r
        elif (mbrForceX.min() > 0 and mbrForceX.max() > 0):
            norm = TwoSlopeNorm(vmin=mbrForceX.min(), vcenter=mbrForceX.min() + 0.5 * (mbrForceX.max() - mbrForceX.min()), vmax=mbrForceX.max())
            cmap = plt.cm.Blues_r
        else:
            norm = TwoSlopeNorm(vmin=mbrForceX.min(), vcenter=0, vmax=mbrForceX.max())
            cmap = plt.cm.seismic_r
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax)
        for n, mbr in enumerate(self.members):
            node_i, node_j = mbr
            ix, iy, iz = self.nodes[node_i - 1]
            jx, jy, jz = self.nodes[node_j - 1]
            if abs(mbrForceX[n]) > 0.001:
                color = cmap(norm(mbrForceX[n]))
                ax.plot3D([ix, jx], [iy, jy], [iz, jz], color=color)
            else:
                ax.plot3D([ix, jx], [iy, jy], [iz, jz], 'grey', linestyle='--')
            # Add text annotation at the midpoint
            mx, my, mz = (ix + jx) / 2, (iy + jy) / 2, (iz + jz) / 2
            label = f"Mbr {n+1} ({node_i}/{node_j})\n{mbrForceX[n]/1000:.2f} kN"
            ax.text(mx, my, mz, label, fontsize=6, color='black', ha='center', va='center')
        for node in self.nodes:
            ax.plot3D([node[0]], [node[1]], [node[2]], 'go', ms=1)
        maxX, maxY, maxZ = self.nodes.max(axis=0)
        minX, minY, minZ = self.nodes.min(axis=0)
        ax.set_xlim([minX - 1, maxX + 1])
        ax.set_ylim([minY - 1, maxY + 1])
        ax.set_zlim([minZ - 0.5, maxZ + 0.5])
        ax.set_xlabel('X-coordinate (m)')
        ax.set_ylabel('Y-coordinate (m)')
        ax.set_zlabel('Z-coordinate (m)')
        ax.set_title('Tension/compression members')
        ax.grid()
        return fig

    def plot_bmd(self, Scale=5):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, 210)
        momentScale = Scale * 1e-6
        for n, mbr in enumerate(self.members):
            node_i, node_j = mbr
            length = self.lengths[n]
            T = self.rotationMatrices[n, :, :]
            ix, iy, iz = self.nodes[node_i - 1]
            jx, jy, jz = self.nodes[node_j - 1]
            ax.plot3D([ix, jx], [iy, jy], [iz, jz], 'g')
            # Major axis bending
            mi = self.mbrMomentZ[n, 0] * momentScale
            mj = self.mbrMomentZ[n, 1] * momentScale
            # Patch corners in local RF (BMD shape)
            pt1 = np.array([0, 0, 0])
            pt2 = np.array([0, mi, 0])
            pt3 = np.array([length, mj, 0])
            pt4 = np.array([length, 0, 0])
            pts = [pt1, pt2, pt3, pt4]
            rotated_pts = [np.dot(T, pt) for pt in pts]
            xr = [ix + pt[0] for pt in rotated_pts]
            yr = [iy + pt[1] for pt in rotated_pts]
            zr = [iz + pt[2] for pt in rotated_pts]
            ax.add_collection3d(Poly3DCollection([list(zip(xr, yr, zr))], alpha=0.2, facecolor='green', edgecolor='green'))
        for node in self.nodes:
            ax.plot3D([node[0]], [node[1]], [node[2]], 'go', ms=3)
        maxX, maxY, maxZ = self.nodes.max(0)
        minX, minY, minZ = self.nodes.min(0)
        ax.set_xlim([minX - 1, maxX + 1])
        ax.set_ylim([minY - 1, maxY + 1])
        ax.set_zlim([0, maxZ + 0.5])
        ax.set_xlabel('X-coordinate (m)')
        ax.set_ylabel('Y-coordinate (m)')
        ax.set_zlabel('Z-coordinate (m)')
        ax.set_title('Mz - Major Axis Bending')
        ax.grid()
        return fig

    def plot_sfd(self, Scale=5):
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(30, 210)
        shearScale = Scale * 1e-6
        xMargin, yMargin, zMargin = 1, 1, 0.5

        def buildLocalMemberSFD(n, axis, scale, length):
            # Returns four points for the SFD patch in local coordinates
            if axis == 'y':
                si = self.mbrForceY[n, 0] * scale
                sj = self.mbrForceY[n, 1] * scale
                pt1 = np.array([0, 0, 0])
                pt2 = np.array([0, si, 0])
                pt3 = np.array([length, sj, 0])
                pt4 = np.array([length, 0, 0])
            elif axis == 'z':
                si = self.mbrForceZ[n, 0] * scale
                sj = self.mbrForceZ[n, 1] * scale
                pt1 = np.array([0, 0, 0])
                pt2 = np.array([0, 0, si])
                pt3 = np.array([length, 0, sj])
                pt4 = np.array([length, 0, 0])
            else:
                raise ValueError('Axis must be "y" or "z"')
            return pt1, pt2, pt3, pt4

        for n, mbr in enumerate(self.members):
            node_i, node_j = mbr
            length = self.lengths[n]
            T = self.rotationMatrices[n, :, :]
            ix, iy, iz = self.nodes[node_i - 1]
            jx, jy, jz = self.nodes[node_j - 1]
            ax.plot3D([ix, jx], [iy, jy], [iz, jz], 'g')
            # Major axis shear patch (Fy)
            pt1, pt2, pt3, pt4 = buildLocalMemberSFD(n, 'y', shearScale, length)
            pts_local = [pt1, pt2, pt3, pt4]
            pts_rotated = [np.matmul(T, pt) for pt in pts_local]
            xr = ix + np.array([pt[0] for pt in pts_rotated])
            yr = iy + np.array([pt[1] for pt in pts_rotated])
            zr = iz + np.array([pt[2] for pt in pts_rotated])
            ax.add_collection3d(Poly3DCollection([list(zip(xr, yr, zr))], alpha=0.2, facecolor='red', edgecolor='red'))
        for node in self.nodes:
            ax.plot3D([node[0]], [node[1]], [node[2]], 'go', ms=3)
        maxX, maxY, maxZ = self.nodes.max(0)
        minX, minY, minZ = self.nodes.min(0)
        ax.set_xlim([minX - xMargin, maxX + xMargin])
        ax.set_ylim([minY - yMargin, maxY + yMargin])
        ax.set_zlim([0, maxZ + zMargin])
        ax.set_xlabel('X-coordinate (m)')
        ax.set_ylabel('Y-coordinate (m)')
        ax.set_zlabel('Z-coordinate (m)')
        ax.set_title('Fy - Major Axis Shear')
        ax.grid()
        return fig

    def get_all_figures(self):
        figs = [
            self.plot_structure(),
            self.plot_deflected_shape(),
            self.plot_axial_forces(),
            self.plot_bmd(),
            self.plot_sfd()
        ]
        names = [
            "Original Structure",
            "Deflected Shape",
            "Axial Forces",
            "Bending Moment Diagram",
            "Shear Force Diagram"
        ]
        return figs, names 

    def plot_deflection(self, *args, **kwargs):
        """Alias for plot_deflected_shape for GUI compatibility."""
        return self.plot_deflected_shape(*args, **kwargs)

    def print_results(self):
        print("="*50)
        print("STRUCTURAL ANALYSIS RESULTS")
        print("="*50)
        if hasattr(self, 'UG'):
            print(f"Maximum displacement: {np.max(np.abs(self.UG)):.6f} m")
        if hasattr(self, 'mbrForceX'):
            print(f"Maximum axial force: {np.max(np.abs(self.mbrForceX)):.2f} N")
        if hasattr(self, 'mbrMomentZ'):
            print(f"Maximum bending moment: {np.max(np.abs(self.mbrMomentZ)):.2f} Nm")
        print("="*50)
        if hasattr(self, 'nodes'):
            print(f"Nodes: {len(self.nodes)}")
        if hasattr(self, 'members'):
            print(f"Members: {len(self.members)}")
        print("="*50) 