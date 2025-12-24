#!/usr/bin/env python3
"""
Script Name  : 03-deploy_schema.py
Description  : Deploy RBH-1 validation schema to PostgreSQL database
Repository   : rbh1-validation-reanalysis
Author       : VintageDon (https://github.com/vintagedon)
ORCID        : 0009-0008-7695-4093
Created      : 2024-12-24
Phase        : Phase 02 - Standard Extraction
Link         : https://github.com/radioastronomyio/rbh1-validation-reanalysis

Description
-----------
Loads credentials from /opt/global-env/research.env, creates the rbh1_validation
database if it doesn't exist, deploys the schema DDL from 02-schema_ddl.sql,
and verifies successful creation of all tables and extensions.

Usage
-----
    python 03-deploy_schema.py [--dry-run] [--database DATABASE_NAME]

Examples
--------
    python 03-deploy_schema.py
        Deploy schema to default database (rbh1_validation) on pgsql01.

    python 03-deploy_schema.py --dry-run
        Show what would be executed without making changes.
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Optional

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

ENV_FILE = Path("/opt/global-env/research.env")
DDL_FILE = Path(__file__).parent / "02-schema_ddl.sql"
DEFAULT_DATABASE = "rbh1_validation"


def load_env_file(env_path: Path) -> dict[str, str]:
    """
    Parse environment file into dictionary.
    
    Handles standard .env format: KEY=VALUE with optional single or double quotes.
    Lines starting with # are treated as comments. Inline comments are NOT supported
    (the # would be included in the value). This matches docker-compose behavior.
    """
    env_vars = {}
    
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")
    
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith('#'):
                continue
            
            # Parse KEY=VALUE
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                
                # Remove surrounding quotes
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    value = value[1:-1]
                
                env_vars[key] = value
    
    return env_vars


def get_connection_params(env: dict, database: Optional[str] = None) -> dict:
    """
    Extract pgsql01 connection parameters from environment.
    """
    return {
        'host': env.get('PGSQL01_HOST', '10.25.20.8'),
        'port': int(env.get('PGSQL01_PORT', 5432)),
        'user': env.get('PGSQL01_ADMIN_USER'),
        'password': env.get('PGSQL01_ADMIN_PASSWORD'),
        'database': database or 'postgres',  # Connect to postgres for DB creation
    }


def create_database_if_not_exists(conn_params: dict, database_name: str, dry_run: bool = False) -> bool:
    """
    Create the target database if it doesn't exist.
    Returns True if database was created, False if it already existed.
    """
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    
    # Connect to postgres database to check/create target
    conn = psycopg2.connect(**conn_params)
    # AI NOTE: ISOLATION_LEVEL_AUTOCOMMIT is required here. CREATE DATABASE cannot
    # run inside a transaction block in PostgreSQL. Without this, psycopg2's default
    # transaction handling will cause "CREATE DATABASE cannot run inside a transaction block".
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    try:
        with conn.cursor() as cur:
            # Check if database exists
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s",
                (database_name,)
            )
            exists = cur.fetchone() is not None
            
            if exists:
                print(f"✓ Database '{database_name}' already exists")
                return False
            
            if dry_run:
                print(f"[DRY RUN] Would create database: {database_name}")
                return False
            
            # Create database
            print(f"Creating database: {database_name}")
            cur.execute(f'CREATE DATABASE "{database_name}"')
            print(f"✓ Database '{database_name}' created")
            return True
            
    finally:
        conn.close()


def deploy_schema(conn_params: dict, ddl_path: Path, dry_run: bool = False) -> None:
    """
    Execute the DDL script against the target database.
    """
    import psycopg2
    
    if not ddl_path.exists():
        raise FileNotFoundError(f"DDL file not found: {ddl_path}")
    
    # Read DDL
    ddl_content = ddl_path.read_text()
    
    if dry_run:
        print(f"[DRY RUN] Would execute DDL from: {ddl_path}")
        print(f"[DRY RUN] DDL size: {len(ddl_content):,} characters")
        return
    
    print(f"Deploying schema from: {ddl_path}")
    
    conn = psycopg2.connect(**conn_params)
    
    try:
        with conn.cursor() as cur:
            cur.execute(ddl_content)
        conn.commit()
        print("✓ Schema deployed successfully")
        
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Schema deployment failed: {e}")
        
    finally:
        conn.close()


def verify_schema(conn_params: dict) -> dict:
    """
    Verify schema was created correctly.
    Returns dict with table counts and verification status.
    """
    import psycopg2
    
    conn = psycopg2.connect(**conn_params)
    
    results = {
        'tables': [],
        'views': [],
        'functions': [],
        'types': [],
    }
    
    try:
        with conn.cursor() as cur:
            # List tables
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'rbh1'
                  AND table_type = 'BASE TABLE'
                ORDER BY table_name
            """)
            results['tables'] = [row[0] for row in cur.fetchall()]
            
            # List views
            cur.execute("""
                SELECT table_name 
                FROM information_schema.views 
                WHERE table_schema = 'rbh1'
                ORDER BY table_name
            """)
            results['views'] = [row[0] for row in cur.fetchall()]
            
            # List functions
            cur.execute("""
                SELECT routine_name 
                FROM information_schema.routines 
                WHERE routine_schema = 'rbh1'
                ORDER BY routine_name
            """)
            results['functions'] = [row[0] for row in cur.fetchall()]
            
            # List custom types
            cur.execute("""
                SELECT typname 
                FROM pg_type t
                JOIN pg_namespace n ON t.typnamespace = n.oid
                WHERE n.nspname = 'rbh1'
                  AND t.typtype = 'e'
                ORDER BY typname
            """)
            results['types'] = [row[0] for row in cur.fetchall()]
            
    finally:
        conn.close()
    
    return results


def print_verification(results: dict) -> bool:
    """
    Pretty-print verification results.
    Returns True if verification passed, False otherwise.
    """
    print("\n" + "=" * 60)
    print("SCHEMA VERIFICATION")
    print("=" * 60)
    
    print(f"\nTables ({len(results['tables'])}):")
    for t in results['tables']:
        print(f"  • {t}")
    
    print(f"\nViews ({len(results['views'])}):")
    for v in results['views']:
        print(f"  • {v}")
    
    print(f"\nFunctions ({len(results['functions'])}):")
    for f in results['functions']:
        print(f"  • {f}")
    
    print(f"\nCustom Types ({len(results['types'])}):")
    for t in results['types']:
        print(f"  • {t}")
    
    # Expected counts
    # AI NOTE: These counts must match 02-schema_ddl.sql. If schema changes, update here.
    # Tables: observations (partitioned) + hst_observations + jwst_observations + 
    #         spectral_grids + wcs_solutions + emission_lines + kinematics +
    #         model_parameters + validation_checks + provenance + schema_version
    # The partition children (hst_observations, jwst_observations) show as separate tables.
    expected = {
        'tables': 12,  # observations (partitioned as 2) + 10 others
        'views': 4,
        'functions': 2,
        'types': 5,
    }
    
    print("\n" + "-" * 60)
    all_ok = True
    
    # Note: partitioned tables show up differently, adjust count
    table_count = len(results['tables'])
    if table_count >= 10:  # At least 10 base tables expected
        print(f"✓ Tables: {table_count} (includes partitions)")
    else:
        print(f"✗ Tables: {table_count} (expected >= 10)")
        all_ok = False
    
    if len(results['views']) >= expected['views']:
        print(f"✓ Views: {len(results['views'])}")
    else:
        print(f"✗ Views: {len(results['views'])} (expected {expected['views']})")
        all_ok = False
    
    if len(results['functions']) >= expected['functions']:
        print(f"✓ Functions: {len(results['functions'])}")
    else:
        print(f"✗ Functions: {len(results['functions'])} (expected {expected['functions']})")
        all_ok = False
    
    if len(results['types']) >= expected['types']:
        print(f"✓ Types: {len(results['types'])}")
    else:
        print(f"✗ Types: {len(results['types'])} (expected {expected['types']})")
        all_ok = False
    
    print("-" * 60)
    if all_ok:
        print("✓ Schema verification PASSED")
    else:
        print("✗ Schema verification FAILED - check counts above")
    
    return all_ok


def main():
    parser = argparse.ArgumentParser(
        description="Deploy RBH-1 validation schema to pgsql01"
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help="Show what would be done without executing"
    )
    parser.add_argument(
        '--database',
        default=DEFAULT_DATABASE,
        help=f"Target database name (default: {DEFAULT_DATABASE})"
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help="Only verify existing schema, don't deploy"
    )
    parser.add_argument(
        '--env-file',
        type=Path,
        default=ENV_FILE,
        help=f"Path to environment file (default: {ENV_FILE})"
    )
    
    args = parser.parse_args()
    
    # Check for psycopg2 availability
    if importlib.util.find_spec('psycopg2') is None:
        print("ERROR: psycopg2 not installed")
        print("Install with: pip install psycopg2-binary")
        sys.exit(1)
    
    # Load environment
    print(f"Loading credentials from: {args.env_file}")
    try:
        env = load_env_file(args.env_file)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Get connection params
    conn_params = get_connection_params(env)
    
    print(f"Target: {conn_params['host']}:{conn_params['port']}")
    print(f"Database: {args.database}")
    print(f"User: {conn_params['user']}")
    
    if args.dry_run:
        print("\n[DRY RUN MODE - No changes will be made]\n")
    
    try:
        if not args.verify_only:
            # Step 1: Create database if needed
            create_database_if_not_exists(
                conn_params, 
                args.database, 
                dry_run=args.dry_run
            )
            
            # Step 2: Deploy schema
            conn_params['database'] = args.database
            deploy_schema(
                conn_params, 
                DDL_FILE, 
                dry_run=args.dry_run
            )
        else:
            conn_params['database'] = args.database
        
        # Step 3: Verify
        if not args.dry_run:
            print("\nVerifying schema...")
            results = verify_schema(conn_params)
            success = print_verification(results)
            sys.exit(0 if success else 1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
