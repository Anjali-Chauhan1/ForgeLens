"use client";
/* eslint-disable react-hooks/purity, react-hooks/immutability --
 * This file is React Three Fiber scene code: mutating objects returned
 * by useThree()/useFrame() every frame, and using Math.random() inside
 * a useMemo(..., []) to freeze one-time random geometry, are the
 * standard, documented R3F patterns (https://r3f.docs.pmnd.rs), not
 * React state mutations. The new React Compiler hook-purity rules are
 * written for plain React state and don't have an R3F exception, so
 * they're disabled for this file rather than fought line by line. */

import { useRef, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Float, MeshDistortMaterial } from "@react-three/drei";
import * as THREE from "three";

const PRIMARY = "#4a3aff";
const VERIFIED = "#0ea968";
const ALERT = "#e8483b";
const CAUTION = "#c9860b";

/** The central "image under analysis" — an organically morphing sphere
 * standing in for a photograph being probed, wrapped by a tilted
 * wireframe ring that reads as an active scan pass. */
function CoreObject() {
  const group = useRef<THREE.Group>(null);
  const ring = useRef<THREE.Mesh>(null);

  useFrame((state, delta) => {
    if (group.current) group.current.rotation.y += delta * 0.18;
    if (ring.current) {
      ring.current.rotation.x += delta * 0.35;
      ring.current.rotation.z += delta * 0.12;
    }
  });

  return (
    <group ref={group}>
      <mesh>
        <icosahedronGeometry args={[1.15, 12]} />
        <MeshDistortMaterial
          color={PRIMARY}
          distort={0.32}
          speed={1.6}
          roughness={0.15}
          metalness={0.3}
          emissive={PRIMARY}
          emissiveIntensity={0.15}
        />
      </mesh>
      <mesh rotation={[Math.PI / 3, 0, 0]}>
        <icosahedronGeometry args={[1.15, 12]} />
        <meshBasicMaterial color={PRIMARY} wireframe transparent opacity={0.12} />
      </mesh>
      <mesh ref={ring} rotation={[Math.PI / 2.4, 0.4, 0]}>
        <torusGeometry args={[1.85, 0.012, 16, 100]} />
        <meshBasicMaterial color={PRIMARY} transparent opacity={0.55} />
      </mesh>
    </group>
  );
}

const FRAGMENTS = [
  { color: VERIFIED, pos: [2.6, 1.1, -0.6] as const, geo: "octahedron", scale: 0.24 },
  { color: ALERT, pos: [-2.4, -0.6, 0.4] as const, geo: "tetrahedron", scale: 0.26 },
  { color: CAUTION, pos: [1.9, -1.5, 0.9] as const, geo: "box", scale: 0.2 },
  { color: PRIMARY, pos: [-2.1, 1.4, -0.3] as const, geo: "octahedron", scale: 0.18 },
  { color: VERIFIED, pos: [0.2, 2.2, -1.1] as const, geo: "tetrahedron", scale: 0.16 },
  { color: ALERT, pos: [-0.4, -2.3, 0.2] as const, geo: "box", scale: 0.15 },
];

function Fragment({ color, pos, geo, scale }: (typeof FRAGMENTS)[number]) {
  const mesh = useRef<THREE.Mesh>(null);
  useFrame((_, delta) => {
    if (mesh.current) {
      mesh.current.rotation.x += delta * 0.4;
      mesh.current.rotation.y += delta * 0.3;
    }
  });

  const geometry =
    geo === "octahedron" ? (
      <octahedronGeometry args={[1, 0]} />
    ) : geo === "tetrahedron" ? (
      <tetrahedronGeometry args={[1, 0]} />
    ) : (
      <boxGeometry args={[1, 1, 1]} />
    );

  return (
    <Float speed={2.4} rotationIntensity={0.6} floatIntensity={1.4}>
      <mesh ref={mesh} position={pos} scale={scale}>
        {geometry}
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.4} roughness={0.3} />
      </mesh>
    </Float>
  );
}

/** Camera drifts toward the cursor position for a subtle parallax feel
 * without giving the user actual orbit controls — this is a showcase
 * object, not something meant to be manually spun. */
function CameraRig() {
  const { camera, pointer } = useThree();
  useFrame(() => {
    camera.position.x += (pointer.x * 0.6 - camera.position.x) * 0.04;
    camera.position.y += (pointer.y * 0.4 - camera.position.y) * 0.04;
    camera.lookAt(0, 0, 0);
  });
  return null;
}

function Particles() {
  const points = useMemo(() => {
    const arr = new Float32Array(120 * 3);
    for (let i = 0; i < 120; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 9;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 9;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 6 - 2;
    }
    return arr;
  }, []);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[points, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#c8c4ff" size={0.02} sizeAttenuation transparent opacity={0.5} />
    </points>
  );
}

export default function ForgeLensScene() {
  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [0, 0, 5.4], fov: 42 }}
      gl={{ antialias: true, alpha: true }}
      className="!touch-auto"
    >
      <ambientLight intensity={0.65} />
      <directionalLight position={[3, 4, 5]} intensity={1.1} />
      <pointLight position={[-3, -2, 2]} intensity={0.6} color={PRIMARY} />

      <CoreObject />
      {FRAGMENTS.map((f, i) => (
        <Fragment key={i} {...f} />
      ))}
      <Particles />
      <CameraRig />
    </Canvas>
  );
}
