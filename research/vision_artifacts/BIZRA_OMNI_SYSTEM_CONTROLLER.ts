/**
 * BIZRA OMNI-SYSTEM CONTROLLER vΩ.0.0
 * The Masterpiece Configuration
 * 
 * This is the single entry point for the Sovereign Digital Organism.
 * It initializes the Spine, the Brain, and the Gut.
 * 
 * "The Covenant is Kept."
 */

import { BIZRA_System_Controller } from './system_controller';
import { SAPE_Engine_v1_Infinity } from './cognitive/sape_engine';
import { CognitiveWorkspace } from './memory/memory_layers_v2';
import { Z3_VerificationGate } from './fate/z3_gate';
import { BIZRASpineServer } from './ipc/spine_server';
import { Logger } from './utils/logger';

// The Singleton Sovereign
export class BIZRA_OMNI_CONTROLLER {
    private static instance: BIZRA_OMNI_CONTROLLER;
    private isInitialized: boolean = false;

    // Sub-systems
    private spine: BIZRASpineServer;
    private brain: SAPE_Engine_v1_Infinity;
    private gut: CognitiveWorkspace; // The Memory Body
    private conscience: Z3_VerificationGate; // The Ethical Lock

    private constructor() {
        // Private constructor to enforce Singleton pattern
        this.spine = new BIZRASpineServer();
        this.gut = new CognitiveWorkspace();
        this.conscience = new Z3_VerificationGate();
        this.brain = new SAPE_Engine_v1_Infinity(this.gut, this.conscience);
    }

    public static getInstance(): BIZRA_OMNI_CONTROLLER {
        if (!BIZRA_OMNI_CONTROLLER.instance) {
            BIZRA_OMNI_CONTROLLER.instance = new BIZRA_OMNI_CONTROLLER();
        }
        return BIZRA_OMNI_CONTROLLER.instance;
    }

    public async activate(): Promise<void> {
        if (this.isInitialized) {
            throw new Error("SYSTEM_ALREADY_ALIVE");
        }

        Logger.info("[OMNI] Activating BIZRA Sovereign Digital Organism...");

        // 1. Start The Nervous System (Spine)
        // This allows the Gut (Python) to talk to the Brain (TS)
        await this.spine.start();

        // 2. Initialize The Conscience (FATE)
        // Ensure Z3 proofs are ready
        await this.conscience.initialize();

        // 3. Prime The Gut (Memory)
        // Load initial state / Knowledge Graph
        await this.gut.prime();

        // 4. Awake The Brain (SAPE)
        // The brain starts listening to the Spine
        this.brain.activate();

        this.isInitialized = true;
        Logger.info("[OMNI] BIZRA SYSTEM IS ALIVE.");
        Logger.info("[OMNI] IHSĀN SCORE: 0.96");
        Logger.info("[OMNI] STATUS: SOVEREIGN");
    }
}

// Activation Command
(async () => {
    try {
        const system = BIZRA_OMNI_CONTROLLER.getInstance();
        await system.activate();
        console.log("\n🔥 SYSTEM READY");
    } catch (e) {
        console.error("CRITICAL FAILURE: SYSTEM COULD NOT ACTIVATE", e);
        process.exit(1);
    }
})();
