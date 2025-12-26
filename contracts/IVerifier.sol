// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * ╔═══════════════════════════════════════════════════════════════════════════════╗
 * ║              BIZRA Ihsān Constitutional Court v1.0.0                          ║
 * ║                   On-Chain ZK Verification Interface                          ║
 * ╠═══════════════════════════════════════════════════════════════════════════════╣
 * ║  This contract verifies Ihsān receipts from the BIZRA sovereignty system:     ║
 * ║    - Single receipt verification (verifyProof)                                ║
 * ║    - Batch/recursive verification (verifyBatchProof)                          ║
 * ║    - Constitution binding enforcement                                          ║
 * ║    - Immutable threshold constants per constitution.toml                      ║
 * ╚═══════════════════════════════════════════════════════════════════════════════╝
 */

/**
 * @title IVerifier
 * @notice Interface for ZK proof verification
 * @dev Implementations should integrate with RiscZero STARK verifier
 */
interface IVerifier {
    /**
     * @notice Verify a single Ihsān receipt ZK proof
     * @param proof The STARK proof bytes from RiscZero prover
     * @param publicInputs Array of public inputs:
     *        [0] agent_id (u64)
     *        [1] transaction_hash (bytes32 as u256)
     *        [2] ihsan_threshold_fixed (u64, default 950)
     *        [3] snr_threshold_fixed (u64, default 750)
     *        [4] constitution_hash (bytes32 as u256)
     * @return success Whether the proof is valid
     */
    function verifyProof(
        bytes calldata proof,
        uint256[5] calldata publicInputs
    ) external view returns (bool success);

    /**
     * @notice Verify a batch/recursive proof for multiple receipts
     * @param proof The aggregated STARK proof bytes
     * @param merkleRoot The Merkle root of the batch
     * @param batchSize Number of receipts in the batch
     * @param constitutionHash Hash of constitution.toml
     * @return success Whether the batch proof is valid
     */
    function verifyBatchProof(
        bytes calldata proof,
        bytes32 merkleRoot,
        uint256 batchSize,
        bytes32 constitutionHash
    ) external view returns (bool success);
}

/**
 * @title IhsanConstitutionalCourt
 * @author BIZRA Sovereignty Framework
 * @notice On-chain settlement layer for Ihsān-verified transactions
 * @dev Implements IVerifier with constitutional threshold enforcement
 * 
 * The Constitutional Court serves as the final arbiter for BIZRA transactions:
 *   1. Verifies ZK proofs that receipts meet Ihsān threshold
 *   2. Enforces constitution.toml binding via hash verification
 *   3. Maintains audit trail via events
 *   4. Supports both single and batch verification
 * 
 * Thresholds are immutable per constitution.toml governance.immutable section.
 */
contract IhsanConstitutionalCourt is IVerifier {
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTANTS (from constitution.toml)
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// @notice Fixed-point Ihsān threshold (0.95 * 1000 = 950)
    /// @dev Immutable per constitution.toml [governance.immutable]
    uint256 public constant IHSAN_THRESHOLD_FIXED = 950;
    
    /// @notice Fixed-point SNR threshold (0.75 * 1000 = 750)
    /// @dev Minimum signal-to-noise ratio for valid receipts
    uint256 public constant SNR_THRESHOLD_FIXED = 750;
    
    /// @notice Fixed-point scale factor
    uint256 public constant FIXED_POINT_SCALE = 1000;
    
    /// @notice Maximum batch size per proof
    /// @dev From constitution.toml [zk.proof.batch_size]
    uint256 public constant MAX_BATCH_SIZE = 1000;
    
    /// @notice RiscZero proof image ID for Ihsān circuit
    /// @dev This must match the compiled circuit image
    bytes32 public immutable CIRCUIT_IMAGE_ID;
    
    /// @notice Expected constitution hash for binding
    bytes32 public immutable CONSTITUTION_HASH;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STATE VARIABLES
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// @notice RiscZero verifier contract address
    address public immutable risc0Verifier;
    
    /// @notice Mapping of settled transaction hashes
    mapping(bytes32 => bool) public settledTransactions;
    
    /// @notice Mapping of settled batch roots
    mapping(bytes32 => BatchSettlement) public settledBatches;
    
    /// @notice Total transactions verified
    uint256 public totalVerified;
    
    /// @notice Total batches settled
    uint256 public totalBatchesSettled;
    
    // ═══════════════════════════════════════════════════════════════════════════
    // STRUCTS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// @notice Settlement record for a batch
    struct BatchSettlement {
        bytes32 merkleRoot;
        uint256 batchSize;
        uint256 settledAt;
        bytes32 proofHash;
    }
    
    /// @notice Decoded receipt for events
    struct DecodedReceipt {
        uint64 agentId;
        bytes32 transactionHash;
        uint64 snrScore;
        uint64 ihsanScore;
        uint64 impactScore;
        uint64 timestamp;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // EVENTS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// @notice Emitted when a single transaction is validated
    /// @param agentId The hashed agent identifier
    /// @param transactionHash Hash of the original envelope
    /// @param ihsanScore Fixed-point Ihsān score
    /// @param blockNumber Block of settlement
    event TransactionValidated(
        uint64 indexed agentId,
        bytes32 indexed transactionHash,
        uint64 ihsanScore,
        uint256 blockNumber
    );
    
    /// @notice Emitted when a batch is settled
    /// @param merkleRoot Root of the receipt Merkle tree
    /// @param batchSize Number of receipts in batch
    /// @param constitutionHash Hash of constitution used
    event BatchSettled(
        bytes32 indexed merkleRoot,
        uint256 batchSize,
        bytes32 constitutionHash
    );
    
    /// @notice Emitted when verification fails
    /// @param reason Failure reason code
    /// @param details Additional context
    event VerificationFailed(
        uint8 indexed reason,
        bytes32 details
    );
    
    // ═══════════════════════════════════════════════════════════════════════════
    // ERRORS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /// @notice Thrown when constitution hash doesn't match
    error ConstitutionMismatch(bytes32 expected, bytes32 actual);
    
    /// @notice Thrown when batch size exceeds maximum
    error BatchSizeExceeded(uint256 size, uint256 maximum);

    /// @notice Thrown when batch size is zero
    error BatchSizeZero();
    
    /// @notice Thrown when transaction already settled
    error AlreadySettled(bytes32 transactionHash);
    
    /// @notice Thrown when proof verification fails
    error ProofVerificationFailed();
    
    /// @notice Thrown when threshold not met
    error ThresholdNotMet(uint256 score, uint256 threshold);
    
    /// @notice Thrown when verifier call fails
    error VerifierCallFailed();
    
    // ═══════════════════════════════════════════════════════════════════════════
    // CONSTRUCTOR
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Deploy the Constitutional Court
     * @param _risc0Verifier Address of RiscZero verifier contract
     * @param _circuitImageId Image ID of the compiled Ihsān circuit
     * @param _constitutionHash Expected constitution.toml hash
     */
    constructor(
        address _risc0Verifier,
        bytes32 _circuitImageId,
        bytes32 _constitutionHash
    ) {
        require(_risc0Verifier != address(0), "Invalid verifier address");
        require(_circuitImageId != bytes32(0), "Invalid circuit image ID");
        require(_constitutionHash != bytes32(0), "Invalid constitution hash");
        
        risc0Verifier = _risc0Verifier;
        CIRCUIT_IMAGE_ID = _circuitImageId;
        CONSTITUTION_HASH = _constitutionHash;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // EXTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Verify a single Ihsān receipt proof
     * @dev Calls RiscZero verifier and checks thresholds
     * @param proof The STARK proof bytes
     * @param publicInputs [agent_id, transaction_hash, ihsan_threshold, snr_threshold, constitution_hash]
     * @return success True if proof is valid and thresholds met
     */
    function verifyProof(
        bytes calldata proof,
        uint256[5] calldata publicInputs
    ) external view override returns (bool success) {
        // Validate constitution hash binding
        if (bytes32(publicInputs[4]) != CONSTITUTION_HASH) {
            return false;
        }
        
        // Validate thresholds match constants
        if (publicInputs[2] != IHSAN_THRESHOLD_FIXED) {
            return false;
        }
        if (publicInputs[3] != SNR_THRESHOLD_FIXED) {
            return false;
        }
        
        // Call RiscZero verifier
        success = _verifyRisc0Proof(proof, abi.encode(publicInputs));
        
        return success;
    }
    
    /**
     * @notice Verify and settle a batch proof
     * @dev Marks batch as settled and emits event
     * @param proof The aggregated STARK proof
     * @param merkleRoot The batch Merkle root
     * @param batchSize Number of receipts
     * @param constitutionHash Constitution binding
     * @return success True if batch verified and settled
     */
    function verifyBatchProof(
        bytes calldata proof,
        bytes32 merkleRoot,
        uint256 batchSize,
        bytes32 constitutionHash
    ) external override returns (bool success) {
        // Validate constitution binding
        if (constitutionHash != CONSTITUTION_HASH) {
            revert ConstitutionMismatch(CONSTITUTION_HASH, constitutionHash);
        }
        
        // Validate batch size
        if (batchSize == 0) {
            revert BatchSizeZero();
        }
        if (batchSize > MAX_BATCH_SIZE) {
            revert BatchSizeExceeded(batchSize, MAX_BATCH_SIZE);
        }
        
        // Check not already settled
        if (settledBatches[merkleRoot].settledAt != 0) {
            revert AlreadySettled(merkleRoot);
        }
        
        // Build public inputs for batch verification
        uint256[4] memory batchInputs = [
            uint256(merkleRoot),
            batchSize,
            IHSAN_THRESHOLD_FIXED,
            uint256(constitutionHash)
        ];
        
        // Verify batch proof
        if (!_verifyRisc0Proof(proof, abi.encode(batchInputs))) {
            revert ProofVerificationFailed();
        }
        
        // Record settlement
        settledBatches[merkleRoot] = BatchSettlement({
            merkleRoot: merkleRoot,
            batchSize: batchSize,
            settledAt: block.timestamp,
            proofHash: keccak256(proof)
        });
        
        totalBatchesSettled++;
        totalVerified += batchSize;
        
        emit BatchSettled(merkleRoot, batchSize, constitutionHash);
        
        return true;
    }
    
    /**
     * @notice Verify and settle a single transaction
     * @param proof The STARK proof
     * @param receipt Encoded receipt data (104 bytes)
     * @return success True if verified and settled
     */
    function settleTransaction(
        bytes calldata proof,
        bytes calldata receipt
    ) external returns (bool success) {
        require(receipt.length == 104, "Invalid receipt length");
        
        // Decode receipt
        DecodedReceipt memory decoded = _decodeReceipt(receipt);
        
        // Check not already settled
        if (settledTransactions[decoded.transactionHash]) {
            revert AlreadySettled(decoded.transactionHash);
        }
        
        // Verify Ihsān threshold
        if (decoded.ihsanScore < IHSAN_THRESHOLD_FIXED) {
            revert ThresholdNotMet(decoded.ihsanScore, IHSAN_THRESHOLD_FIXED);
        }
        
        // Verify SNR threshold
        if (decoded.snrScore < SNR_THRESHOLD_FIXED) {
            revert ThresholdNotMet(decoded.snrScore, SNR_THRESHOLD_FIXED);
        }
        
        // Build public inputs
        uint256[5] memory publicInputs = [
            uint256(decoded.agentId),
            uint256(decoded.transactionHash),
            IHSAN_THRESHOLD_FIXED,
            SNR_THRESHOLD_FIXED,
            uint256(CONSTITUTION_HASH)
        ];
        
        // Verify proof
        if (!_verifyRisc0Proof(proof, abi.encode(publicInputs))) {
            revert ProofVerificationFailed();
        }
        
        // Mark as settled
        settledTransactions[decoded.transactionHash] = true;
        totalVerified++;
        
        emit TransactionValidated(
            decoded.agentId,
            decoded.transactionHash,
            decoded.ihsanScore,
            block.number
        );
        
        return true;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // VIEW FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Check if a transaction has been settled
     * @param transactionHash The transaction hash to check
     * @return True if settled
     */
    function isSettled(bytes32 transactionHash) external view returns (bool) {
        return settledTransactions[transactionHash];
    }
    
    /**
     * @notice Check if a batch has been settled
     * @param merkleRoot The batch Merkle root
     * @return True if settled
     */
    function isBatchSettled(bytes32 merkleRoot) external view returns (bool) {
        return settledBatches[merkleRoot].settledAt != 0;
    }
    
    /**
     * @notice Get batch settlement details
     * @param merkleRoot The batch Merkle root
     * @return settlement The settlement record
     */
    function getBatchSettlement(bytes32 merkleRoot) 
        external 
        view 
        returns (BatchSettlement memory settlement) 
    {
        return settledBatches[merkleRoot];
    }
    
    /**
     * @notice Validate thresholds without proof (for testing)
     * @param ihsanScore The Ihsān score (fixed-point)
     * @param snrScore The SNR score (fixed-point)
     * @return valid True if both thresholds met
     */
    function validateThresholds(
        uint256 ihsanScore,
        uint256 snrScore
    ) external pure returns (bool valid) {
        return ihsanScore >= IHSAN_THRESHOLD_FIXED && 
               snrScore >= SNR_THRESHOLD_FIXED;
    }
    
    // ═══════════════════════════════════════════════════════════════════════════
    // INTERNAL FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════
    
    /**
     * @notice Decode a 104-byte receipt
     * @param data The encoded receipt
     * @return decoded The decoded receipt struct
     */
    function _decodeReceipt(bytes calldata data) 
        internal 
        pure 
        returns (DecodedReceipt memory decoded) 
    {
        // Format (little-endian):
        // - agent_id: u64 (8 bytes)
        // - transaction_hash: [u8; 32] (32 bytes)
        // - snr_score: u64 (8 bytes)
        // - ihsan_score: u64 (8 bytes)
        // - impact_score: u64 (8 bytes)
        // - timestamp: u64 (8 bytes)
        // - nonce: [u8; 32] (32 bytes)
        
        decoded.agentId = _readUint64LE(data, 0);
        decoded.transactionHash = bytes32(data[8:40]);
        decoded.snrScore = _readUint64LE(data, 40);
        decoded.ihsanScore = _readUint64LE(data, 48);
        decoded.impactScore = _readUint64LE(data, 56);
        decoded.timestamp = _readUint64LE(data, 64);
        // nonce at bytes 72-104 (not decoded into struct)
    }
    
    /**
     * @notice Read uint64 in little-endian format
     * @param data The byte array
     * @param offset Starting offset
     * @return value The decoded uint64
     */
    function _readUint64LE(bytes calldata data, uint256 offset) 
        internal 
        pure 
        returns (uint64 value) 
    {
        assembly {
            // Load 8 bytes from calldata at offset
            let ptr := add(data.offset, offset)
            let raw := calldataload(ptr)
            // Extract the first 8 bytes at offset (big-endian word load)
            value := shr(192, raw)
        }
        // Byte swap for little-endian
        value = _swapBytes64(value);
    }
    
    /**
     * @notice Swap bytes of uint64 (big-endian <-> little-endian)
     */
    function _swapBytes64(uint64 value) internal pure returns (uint64) {
        return ((value & 0xFF) << 56) |
               ((value & 0xFF00) << 40) |
               ((value & 0xFF0000) << 24) |
               ((value & 0xFF000000) << 8) |
               ((value & 0xFF00000000) >> 8) |
               ((value & 0xFF0000000000) >> 24) |
               ((value & 0xFF000000000000) >> 40) |
               ((value & 0xFF00000000000000) >> 56);
    }
    
    /**
     * @notice Verify proof using RiscZero verifier
     * @dev Placeholder for actual RiscZero integration
     * @param proof The STARK proof
     * @param publicInputs The public inputs
     * @return valid True if proof is valid
     */
    function _verifyRisc0Proof(
        bytes calldata proof,
        bytes memory journal
    ) internal view returns (bool valid) {
        if (proof.length == 0) {
            return false;
        }

        try IRiscZeroVerifier(risc0Verifier).verify(
            CIRCUIT_IMAGE_ID,
            proof,
            journal
        ) returns (bool ok) {
            return ok;
        } catch {
            revert VerifierCallFailed();
        }
    }
}


/**
 * @title IRiscZeroVerifier
 * @notice Interface for RiscZero STARK verifier
 * @dev To be integrated when RiscZero contracts are deployed
 */
interface IRiscZeroVerifier {
    /**
     * @notice Verify a RiscZero STARK proof
     * @param imageId The circuit image ID
     * @param proof The proof bytes
     * @param journal The public outputs (journal)
     * @return True if proof is valid
     */
    function verify(
        bytes32 imageId,
        bytes calldata proof,
        bytes calldata journal
    ) external view returns (bool);
}
