import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
from qiskit import QuantumCircuit

qc = QuantumCircuit(4)
for q in range(4):
    qc.rx(0.1, q)
    qc.ry(0.2, q)
    qc.rz(0.3, q)
for q in range(3):
    qc.cx(q, q + 1)

print("\n=== IMPLEMENTED PARAMETERIZED QUANTUM CIRCUIT (LAYER 1) ===")
print(qc.draw(output="text", fold=-1))
print("============================================================\n")
